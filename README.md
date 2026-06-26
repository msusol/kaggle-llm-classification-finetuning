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
| Current best | _none yet — see [docs/plans/leaderboard.md](docs/plans/leaderboard.md)_ |
| Target tier | top public solutions ~0.89–0.92 (Gemma-2-9b QLoRA) |

## Evaluator constraints

This is a **Code Competition**. The notebook must satisfy:

| Constraint | Value | Implication |
|---|---|---|
| Runtime | ≤ 9 h GPU | Quantized / batched inference over ~25k test rows |
| Internet | disabled | Bundle base weights + adapter + wheels as Kaggle datasets |
| Output | `submission.csv` | `id,winner_model_a,winner_model_b,winner_tie` probabilities |
| External data | allowed | Pretrained models + extra arena data permitted |

## Hardware

Training on **NVIDIA DGX Spark (GB10)** — reuse the working LoRA/PEFT setup from
`mineral-hr-llm` on `sparkdb62`. See
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
