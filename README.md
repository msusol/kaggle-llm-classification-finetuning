# LLM Classification Finetuning — Chatbot Arena Human Preference

https://www.kaggle.com/competitions/llm-classification-finetuning

Predict which of two anonymous LLM responses a human judge preferred in a Chatbot
Arena head-to-head. Three-class target (`winner_model_a`, `winner_model_b`,
`winner_tie`), scored on multiclass **log loss**.

## Goal

Train a sequence-classification head on a mid-size instruction-tuned LLM
(QLoRA, bf16) over the concatenated `prompt + response_a + response_b`, then ship a
**fully offline** Kaggle notebook that loads the bundled adapter + base weights and
writes `submission.csv`.

## Status

| Item | Value |
|---|---|
| Baseline to beat (uniform 0.333) | log loss ≈ **1.0986** (ln 3) |
| v0.1 TF-IDF + LightGBM | **1.04157** (LB) |
| v0.2 Llama-3.2-3B LoRA | _in progress_ |
| Target tier | top public solutions ~0.89–0.92 |

## Phases

### What each version does

| Version | Notebook | Technique | PEFT? | Notes |
|---|---|---|---|---|
| v0.1 | `v0.1-tfidf-baseline` | TF-IDF + LightGBM | No | CPU only; locks the submission pipeline |
| v0.2 | `v0.2-llama-qlora` | Llama-3.2-3B SFT + LoRA | Yes | Single-fold validation run on Kaggle T4 |

### What PEFT/LoRA means here

**PEFT** (Parameter-Efficient Fine-Tuning) via **LoRA** means we inject small trainable
adapter matrices into the frozen base model's attention layers and train only those
(~24M of 3.2B params, < 1%). The base model weights never change.

This is **not** teacher/student / knowledge distillation. There is no second "teacher"
model. We fine-tune Llama-3.2-3B directly on the human preference labels from
`train.csv`. At inference, instead of generating text, we read the model's logits at
the final token position for the tokens `A`, `B`, and `tie` and softmax them into
3-class probabilities.

### Ensemble strategy (DeBERTa optional)

The original plan included a DeBERTa-v3-large rung between the baseline and the LLM.
DeBERTa is optional — the ensemble still works without it:

**Without DeBERTa:**
- Average predictions across all 5 LoRA folds (same architecture, different held-out
  fold each time) — the primary diversity signal
- Optionally add a second model size (e.g. Llama-3.1-8B) if quota allows
- Swap-TTA is already baked into every fold (average forward + swapped predictions)

**With DeBERTa (stronger):**
- DeBERTa fold-ensemble + Llama fold-ensemble, weighted average
- DeBERTa converges faster and gives a different inductive bias (cross-encoder vs.
  generative label readout), so ensembling the two is worth ~0.01–0.02 log loss

The LLM fold-ensemble alone is sufficient to reach the top tier.

**True distillation (< 0.82 target):**

The winning approach [cite:2] used **knowledge distillation**, which is distinct from
what v0.2 does:

- **Teacher:** a frozen 70B model runs inference over the full training set and
  produces soft probability distributions (not hard A/B/tie labels).
- **Student:** per-fold Gemma-2-9b fine-tuned via LoRA to match those soft targets
  (KL-divergence loss against the teacher's output, not cross-entropy against the
  human labels).
- **Weight averaging:** the per-fold LoRA adapters are averaged before submission.

This is PEFT on the student side — the teacher is never trained, only used for
inference. The student learns a richer signal (the teacher's uncertainty over all three
classes) than the binary human vote alone provides. This is what separates the ~0.89
tier (SFT on hard labels) from the < 0.82 tier (distillation from a 70B teacher).

A distillation phase (v0.3 or v0.4) would require:
1. Running 70B inference over the full training set (DGX Spark, offline batch)
2. Saving soft targets as a dataset
3. Swapping the SFT loss for KL-divergence against those targets

## Evaluator constraints

This is a **Code Competition**. The notebook must satisfy:

| Constraint | Value | Implication |
|---|---|---|
| Runtime | ≤ 9 h GPU | Quantized / batched inference over ~25k test rows |
| Internet | disabled | Bundle base weights + adapter + wheels as Kaggle datasets |
| Output | `submission.csv` | `id,winner_model_a,winner_model_b,winner_tie` probabilities |
| External data | allowed | Pretrained models + extra arena data permitted |

## Hardware

Training on **NVIDIA DGX Spark (GB10)** by reusing the verified LoRA/QLoRA harness in
[`msusol/mineral-hr-llm`](https://github.com/msusol/mineral-hr-llm) on `sparkdb62`
(`train_hr_lora.py`, `Dockerfile.gb10`, `--use_qlora`, `--local_files_only`). See
[docs/plans/v0.2-gemma-qlora-plan.md](docs/plans/v0.2-gemma-qlora-plan.md).

## Data

| File | Rows | Columns |
|---|---|---|
| `train.csv` | ~55,000 | `id, model_a, model_b, prompt, response_[a/b], winner_model_[a/b/tie]` |
| `test.csv` | ~25,000 (hidden) | `id, prompt, response_[a/b]` |
| `sample_submission.csv` | — | `id, winner_model_[a/b/tie]` |

> `model_a` / `model_b` exist in train but **not** test — never use them as model
> features (leakage). They are valid for CV stratification and analysis only.

## Layout

```
docs/
  plans/        competition-overview, implementation-plan, versioned vX.Y plans, TODO, leaderboard, CITATIONS, submission-checklist
  adr/          architecture decision records (NNNN-title.md)
  investigate/  investigation logs
  images/       plots, dashboards
scripts/        download_data.sh, train_lora.py, infer_submission.py, package_submission.sh
notebook/       Kaggle kernel .ipynb + kernel-metadata.json
configs/        training YAML
data/           train.csv / test.csv (gitignored); generated splits gitignored
```

## Quick start

```bash
# 1. Auth + download (needs ~/.kaggle/kaggle.json and accepted comp rules)
bash scripts/download_data.sh

# 2. EDA + baseline (rung 1)
#    see docs/plans/v0.1-baseline-plan.md

# 3. Gemma-2-9b QLoRA on DGX Spark (rung 3)
#    see docs/plans/v0.2-gemma-qlora-plan.md
```

## Strategy summary

See [docs/plans/implementation-plan.md](docs/plans/implementation-plan.md) for the
full ladder. Short version: lock the offline submission pipeline with a cheap
baseline first, then climb to Gemma-2-9b QLoRA, then ensemble / swap-TTA.
