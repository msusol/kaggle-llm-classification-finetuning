# Leaderboard

Update after **every** completed run + validation pass. OOF = out-of-fold CV log loss.

| Version | Model | Key change | OOF log loss | Kaggle LB | Notes |
|---|---|---|---|---|---|
| uniform | — | 0.333 each | 1.0986 | — | must-beat floor (ln 3) |
| v0.1 | LightGBM | TF-IDF + length feats | **1.0404** | _tbd_ | beats uniform; needs notebook submission for LB |
| v0.2 | gemma-2-9b QLoRA | seq-cls, swap-TTA | _tbd_ | _tbd_ | target ~0.89–0.92 |

## Run log

### 2026-06-26 — v0.1 LightGBM baseline

- **Config:** TF-IDF word unigrams+bigrams (50k features) + 7 length features; LightGBM multiclass, num_leaves=63, lr=0.05, 5-fold stratified
- **OOF log loss:** 1.0404 (folds: 1.0414, 1.0421, 1.0385, 1.0383, 1.0418)
- **Best iter per fold:** ~62–80 (early stopping at patience=50)
- **Kaggle LB:** _pending notebook submission_
- **Takeaway:** beats uniform (1.0986) as expected; purely plumbing validation. LB score requires a Kaggle notebook to access the hidden test set.
