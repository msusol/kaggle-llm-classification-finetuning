# TODO

## Phase 0 — Setup
- [ ] Place `~/.kaggle/kaggle.json`; accept competition rules on the website
- [ ] `bash scripts/download_data.sh` → data/train.csv, test.csv, sample_submission.csv
- [ ] Confirm uniform submission scores ≈ 1.0986 (sanity)

## Phase 1 — Baseline (v0.1)
- [ ] EDA: class balance, length distributions, position-bias check
- [ ] TF-IDF + length features → LightGBM 5-fold
- [ ] First offline submission; record CV/LB in leaderboard.md

## Phase 2 — DeBERTa-v3 (v0.x, optional checkpoint)
- [ ] 3-class cross-encoder; confirm < 9 h inference
- [ ] Record OOF + LB

## Phase 3 — Gemma-2-9b QLoRA (v0.2)
- [ ] Fork mineral-hr-llm LoRA harness on sparkdb62 → seq-classification head (3 labels)
- [ ] A/B swap augmentation + tail truncation
- [ ] 5-fold train on GB10; select by OOF log loss
- [ ] Package offline (base + adapter + wheels) with swap-TTA inference
- [ ] Submit; update leaderboard.md

## Phase 4 — Squeeze
- [ ] Ensemble DeBERTa + Gemma
- [ ] Tie-class calibration
- [ ] (Stretch) distill larger teacher into Gemma students
