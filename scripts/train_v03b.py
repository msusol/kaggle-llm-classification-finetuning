#!/usr/bin/env python3
import os, gc, torch, numpy as np, pandas as pd, multiprocessing
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
os.environ["PEFT_AUTOCAST_ADAPTER_DTYPE"] = "0"

MAX_SEQ_LEN = 1024
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
LORA_R = 16
NUM_EPOCHS = 1
SEED = 42
OUTPUT_DIR = Path("output/v0.3b_adapters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = "google/gemma-2-2b-it"

def build_prompt(row, swap=False):
    prompt = f"Prompt:\n{row['prompt']}\n\n"
    if not swap:
        prompt += f"Response A:\n{row['response_a']}\n\nResponse B:\n{row['response_b']}\n\n"
    else:
        prompt += f"Response A:\n{row['response_b']}\n\nResponse B:\n{row['response_a']}\n\n"
    prompt += "Which response is better? Answer with 'A', 'B', or 'tie'."
    return prompt

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize_fn(examples):
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": a}], tokenize=False) for p, a in zip(examples["prompt_text"], examples["answer"])]
    return tokenizer(texts, padding=False, truncation=True, max_length=MAX_SEQ_LEN)

def prepare_fold(df, fold, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    labels = df["label"].values
    train_idx, val_idx = list(skf.split(df, labels))[fold]
    df_train = df.iloc[train_idx].copy()
    train_rows = []
    for _, row in df_train.iterrows():
        ans_idx = row["label"]
        if ans_idx == 0: ans_fwd, ans_swp = "A", "B"
        elif ans_idx == 1: ans_fwd, ans_swp = "B", "A"
        else: ans_fwd, ans_swp = "tie", "tie"
        train_rows.append({"prompt_text": build_prompt(row, swap=False), "answer": ans_fwd})
        train_rows.append({"prompt_text": build_prompt(row, swap=True), "answer": ans_swp})
    raw_ds = Dataset.from_list(train_rows)
    return raw_ds.map(tokenize_fn, batched=True, batch_size=1000, num_proc=multiprocessing.cpu_count(), remove_columns=raw_ds.column_names, desc="Tokenizing")

CHECKPOINT_DIR = Path("output/v0.3b_checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_STEPS = 200

def _latest_checkpoint(fold_idx):
    ckpt_dir = CHECKPOINT_DIR / f"fold_{fold_idx}"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None

def train_fold(fold_idx, df):
    adapter_dir = OUTPUT_DIR / f"fold_{fold_idx}"
    if (adapter_dir / "adapter_config.json").exists():
        print(f"\n--- Fold {fold_idx} already complete, skipping ---")
        return

    print(f"\n--- Training Fold {fold_idx} ---")
    train_ds = prepare_fold(df, fold_idx)
    total_steps = (len(train_ds) // BATCH_SIZE // GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps = max(1, int(0.05 * total_steps))

    resume_from = _latest_checkpoint(fold_idx)
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2")
    peft_config = LoraConfig(r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(CHECKPOINT_DIR / f"fold_{fold_idx}"),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=25,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=2,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            seed=SEED,
            report_to="none",
            dataset_text_field="",
            max_length=MAX_SEQ_LEN,
        ),
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(adapter_dir))
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df["label"] = df[["winner_model_a", "winner_model_b", "winner_tie"]].values.argmax(axis=1)
    for i in range(5): train_fold(i, df)
