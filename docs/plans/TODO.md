# TODO

## Phase 0 — Setup
- [x] Place `~/.kaggle/kaggle.json`; accept competition rules on the website
- [x] `bash scripts/download_data.sh` → data/train.csv, test.csv, sample_submission.csv
- [x] Confirm uniform submission scores ≈ 1.0986 (sanity)

## Phase 1 — Baseline (v0.1)
- [ ] EDA: class balance, length distributions, position-bias check
- [x] TF-IDF + length features → LightGBM 5-fold (OOF 1.0404)
- [ ] Build Kaggle notebook for baseline inference → `kaggle kernels push` → record LB

## Phase 2 — DeBERTa-v3 (v0.x, optional checkpoint)
- [ ] 3-class cross-encoder; confirm < 9 h inference
- [ ] Record OOF + LB

## Phase 3 — QLoRA via mineral-hr-llm (v0.2)
- [ ] `scripts/convert_to_jsonl.py`: Kaggle CSV → {id,instruction,context,output} + A/B swap aug
- [ ] v0.2a (Kaggle T4×2): train Llama-3.1-8B LoRA in Kaggle notebook (internet on); validate pipeline end-to-end; submit directly
- [ ] v0.2b (DGX Spark): scale on GB10 (Gemma-2-9b or more epochs); push adapter to HF → Kaggle Models → offline submission
- [ ] Package offline (base + adapter + source-built bitsandbytes) with swap-TTA inference
- [ ] Submit; update leaderboard.md

## Phase 4 — Squeeze
- [ ] Ensemble DeBERTa + Gemma
- [ ] Tie-class calibration
- [ ] (Stretch) distill larger teacher into Gemma students
