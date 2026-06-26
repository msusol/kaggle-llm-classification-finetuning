# Leaderboard

Update after **every** completed run + validation pass. OOF = out-of-fold CV log loss.

| Version | Model | Key change | OOF log loss | Kaggle LB | Notes |
|---|---|---|---|---|---|
| uniform | — | 0.333 each | 1.0986 | — | must-beat floor (ln 3) |
| v0.1 | LightGBM | TF-IDF + length feats | _tbd_ | _tbd_ | plumbing baseline |
| v0.2 | gemma-2-9b QLoRA | seq-cls, swap-TTA | _tbd_ | _tbd_ | target ~0.89–0.92 |

## Run log

_(append dated entries: config, OOF, LB, takeaway)_
