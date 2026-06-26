# Competition Overview

## Objective

Predict which of two anonymous LLM responses a human judge preferred in a Chatbot
Arena head-to-head battle.[cite:1] For each test `id`, output a probability for each
of three classes: `winner_model_a`, `winner_model_b`, `winner_tie`.[cite:1]

## Metric

Multiclass **log loss** between predicted probabilities and ground truth, with
`eps=auto`.[cite:1] The uniform prediction (0.333 each) gives log loss ≈ **1.0986**
(ln 3) — this is the must-beat floor.

## Data

| File | Rows | Columns |
|---|---|---|
| `train.csv` | ~55,000 | `id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie` |
| `test.csv` | ~25,000 (hidden) | `id, prompt, response_a, response_b` |
| `sample_submission.csv` | — | `id, winner_model_a, winner_model_b, winner_tie` |

Notes:[cite:1]
- `model_a` / `model_b` appear in train but **not** test → leakage risk; do not use as features.
- `prompt` may contain multiple turns; `response_[a/b]` are the two models' answers.
- The dataset may contain profane / offensive text — do not filter it; it is in-distribution.

## Code competition requirements

- CPU **or** GPU notebook ≤ **9 hours** run-time.[cite:1]
- **Internet access disabled** during scoring.[cite:1]
- Freely & publicly available external data + pretrained models are allowed.[cite:1]
- Submission file must be named **`submission.csv`**.[cite:1]
- Runtimes are slightly obfuscated (±15 min variance).[cite:1]

## Format & timeline

Getting Started competition with a **rolling 2-month leaderboard** — submissions older
than two months are invalidated; a team with no submission in the prior two months
drops off.[cite:1] No cash prize; indefinite timeline.[cite:1]

## Relationship to LMSYS competition

This uses the same Chatbot Arena data and identical 3-class format as the featured
**LMSYS – Chatbot Arena Human Preference Predictions** competition ($100k).[cite:2]
That competition's published solutions are the playbook here — see
[implementation-plan.md](implementation-plan.md).
