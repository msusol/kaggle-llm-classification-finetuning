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

## Phase 3 — QLoRA via mineral-hr-llm (v0.2)
- [ ] Check out msusol/mineral-hr-llm on sparkdb62; build Dockerfile.gb10
- [ ] `scripts/convert_to_jsonl.py`: Kaggle CSV → {id,instruction,context,output} + A/B swap aug
- [ ] v0.2a: train Llama-3.1-8B with run_train.sh (label-token readout); OOF log loss
- [ ] v0.2b: repoint --model_name to local gemma-2-9b-it; compare OOF
- [ ] Package offline (base + adapter + source-built bitsandbytes) with swap-TTA inference
- [ ] Submit; update leaderboard.md

## Phase 4 — Squeeze
- [ ] Ensemble DeBERTa + Gemma
- [ ] Tie-class calibration
- [ ] (Stretch) distill larger teacher into Gemma students
