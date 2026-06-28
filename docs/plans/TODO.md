# TODO

## Phase 0 — Setup
- [x] Place `~/.kaggle/kaggle.json`; accept competition rules on the website
- [x] `bash scripts/download_data.sh` → data/train.csv, test.csv, sample_submission.csv
- [x] Confirm uniform submission scores ≈ 1.0986 (sanity)

## Phase 1 — Baseline (v0.1)
- [ ] EDA: class balance, length distributions, position-bias check — separate notebook (`v0.0-eda.ipynb`)
- [x] TF-IDF + length features → LightGBM 5-fold (OOF 1.0404)
- [ ] Push v0.1 baseline notebook and record LB — deferred; jumped directly to LLM approach

## Phase 2 — DeBERTa-v3
- deferred — skipped as standalone phase; DeBERTa retained as optional component in Phase 5 ensemble

## Phase 3 — LLM SFT + LoRA

### v0.2 — Llama-3.2-3B on TPU v5e-8 (pipeline validation)
- [x] v0.2a (Kaggle T4): Llama-3.2-3B fp16 LoRA — running (v24, all P100 workarounds removed)
- [ ] Record fold 0 OOF once v24 completes; update leaderboard.md
- [ ] Submit v0.2 and record LB in leaderboard.md

### v0.3 — Gemma-2-9b on TPU v5e-8
- [ ] Attach `google/gemma-2/transformers/gemma-2-9b-it/1` via Kaggle UI (license)
- [ ] Push `v0.3-gemma-tpu` notebook; stop auto-run; select TPU v5e-8 in UI; run
- [ ] Validate fold 0 OOF log loss < 0.95; record in leaderboard.md
- [ ] If pipeline stable, run all 5 folds and average adapters
- [ ] Submit v0.3 and record LB

### v0.3b — Gemma-2-9b on DGX Spark GB10 (bf16, all 5 folds)
- [ ] Train all 5 folds in bf16 on DGX (no quantization)
- [ ] Write `scripts/average_lora.py` — merge fold adapters by averaging LoRA A/B matrices
- [ ] Run averaging script → single merged adapter (~200 MB)
- [ ] Upload merged adapter as private Kaggle dataset (`gdataranger/gemma-2-9b-lora-v03b`)
- [ ] Create `notebook/v0.3b-infer.ipynb` + `v0.3b-infer-kernel-metadata.json` (offline, base + adapter, no GPU required at push)
- [ ] Push v0.3b-infer; stop auto-run; run with GPU T4; write submission.csv
- [ ] Record OOF + LB in leaderboard.md

### v0.4 — Reward model zero-shot (Skywork-Reward-Llama-3.1-8B)
- [ ] Attach `quincyqiang/skywork-reward-llama-3.1-8b-v0.2` via Kaggle UI
- [ ] Score response_a and response_b independently → compute score delta
- [ ] Synthesize tie probability from score proximity; calibrate temperature + tie_weight on OOF
- [ ] Record OOF log loss vs v0.3; apply decision gate (see v0.4 plan)
- [ ] If competitive: try v0.4b on DGX with Skywork-Reward-Gemma-2-27B (zero-shot, 27B quality)

## Phase 4 — Distillation (< 0.82 target)
- [ ] Run `Skywork/Skywork-Reward-Gemma-2-27B` teacher on DGX → save soft labels
- [ ] Train `gemma-2-9b-it` student with LoRA + KL-divergence loss (5 folds)
- [ ] Average LoRA layers across folds; upload adapter; submit

## Phase 5 — Squeeze
- [ ] Ensemble v0.3 (SFT folds) + Phase 4 (distilled folds)
- [ ] Optionally add DeBERTa-v3-large cross-encoder for diversity
- [ ] Tie-class calibration
