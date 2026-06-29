# Leaderboard

Update after **every** completed run + validation pass. OOF = out-of-fold CV log loss.

| Version | Model | Key change | OOF log loss | Kaggle LB | Notes |
|---|---|---|---|---|---|
| uniform | — | 0.333 each | 1.0986 | — | must-beat floor (ln 3) |
| v0.1 | LightGBM | TF-IDF + length feats | **1.0404** | **1.04157** | CV/LB near-perfect correlation; pipeline confirmed |
| v0.2 | gemma-2-9b QLoRA | seq-cls, swap-TTA | _tbd_ | _tbd_ | target ~0.89–0.92 |
| v0.3b | gemma-2-2b-it LoRA | causal-LM, swap-aug x2, 5-fold | _pending inference_ | _tbd_ | fold 0: train_loss=0.993, acc=0.745; folds 1–4 in progress with early stopping |
| v0.4a | Skywork-Reward-Llama-3.1-8B | reward zero-shot, swap-TTA, temp=0.5, tie=0.5 | **1.5828** (calib n=1000) | — | 0.54 worse than baseline; decision gate → skip v0.4c, proceed to Phase 4 |

## Run log

### 2026-06-28 — v0.3b gemma-2-2b-it LoRA 5-fold (in progress)

- **Config:** gemma-2-2b-it, LoRA r=16 α=32, causal-LM on chat template, swap augmentation (2× train), MAX_SEQ_LEN=1024, bf16, flash_attention_2, lr=2e-4 cosine, 1 epoch per fold, batch=4×4 grad accum
- **Fold 0:** train_loss=0.9928, mean_token_accuracy=0.7454, epoch=1.0 (full epoch, old script — no early stopping or val eval)
- **Folds 1–4:** running with EarlyStoppingCallback (patience=3, eval_steps=200), val = held-out fold forward-pass only
- **OOF log loss:** _pending_ — requires inference pass after all 5 adapters saved
- **Kaggle LB:** _tbd_
- **Takeaway:** In progress. Fold 1 now training (~21h per fold without early stopping; expect shorter with patience).

### 2026-06-28 — v0.4a Skywork reward model zero-shot

- **Config:** Skywork-Reward-Llama-3.1-8B-v0.2 (`shelterw/skywork`), bf16, sdpa, MAX_SEQ_LEN=1024; score_response per (prompt, response) pair; sigmoid delta → 3-class probs; swap-TTA; temperature=0.5, tie_weight=0.5 (grid search on CALIB_SAMPLE=1000 from fold 0 val)
- **Calib OOF log loss:** 1.5828 (n=1,000 subsample of fold 0 val, swap-TTA)
- **Kaggle LB:** not submitted (decision gate triggered)
- **Takeaway:** 0.54 worse than TF-IDF baseline. Zero-shot reward scoring does not transfer well to this preference task without fine-tuning. Decision gate: >0.10 worse than baseline → skip v0.4c; proceed to Phase 4 (distillation).

### 2026-06-26 — v0.1 LightGBM baseline

- **Config:** TF-IDF word unigrams+bigrams (50k features) + 7 length features; LightGBM multiclass, num_leaves=63, lr=0.05, 5-fold stratified
- **OOF log loss:** 1.0404 (folds: 1.0414, 1.0421, 1.0385, 1.0383, 1.0418)
- **Best iter per fold:** ~62–80 (early stopping at patience=50)
- **Kaggle LB:** 1.04157
- **Takeaway:** beats uniform (1.0986); CV (1.0404) and LB (1.04157) near-identical — no leakage, no distribution shift. Pipeline confirmed end-to-end.
