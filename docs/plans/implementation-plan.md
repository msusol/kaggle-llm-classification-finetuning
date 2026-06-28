# Implementation Plan — Attack Strategy

The strategy is a **ladder**: each rung produces a valid `submission.csv` and a CV
score, so we always have a working offline pipeline before adding model complexity.
The pipeline plumbing (offline weights, 9 h runtime budget) is where competitors lose
time — not the modeling.

## Known winning shape

Top LMSYS solutions converged on the same recipe: finetune a mid-size
instruction-tuned LLM as a **3-class sequence classifier** over the concatenated
`prompt + response_a + response_b`, with LoRA/QLoRA in bf16.[cite:2][cite:3] The
consensus backbone was `gemma-2-9b-it` (beat Llama-3/3.1-8b).[cite:3][cite:4] The
top team distilled a 70B teacher into per-fold Gemma-2-9b students and averaged the
LoRA weights.[cite:2]

## The ladder

### Rung 0 — Pipeline skeleton
- `scripts/download_data.sh` pulls the competition data.
- Produce a uniform-probability `submission.csv` and confirm it scores ≈ 1.0986.
- **Exit criterion:** a notebook that runs offline and emits a valid submission.

### Rung 1 — Cheap baseline (CPU)
- Features: TF-IDF over `response_a` / `response_b`, length & length-ratio, token
  overlap with prompt, simple verbosity/position features.
- Model: logistic regression or LightGBM, multinomial, 3-class.
- **Expected:** ~1.00–1.03 log loss. Confirms CV harness + submission format.

### Rung 2 — DeBERTa-v3-large (GPU, fast)
- Cross-encoder: 3-class head over `[prompt] [resp_a] [resp_b]`.
- Fits comfortably in 9 h; good checkpoint before committing to a 9B model.
- **Expected:** ~0.95–1.00.

### Rung 3 — LLM SFT + LoRA (the contender)
- Label-token readout: softmax over logits at the final token position for tokens A/B/tie.
- Swap-TTA baked in: average forward + A↔B-swapped predictions per example.
- **v0.2a (Kaggle T4):** Llama-3.2-3B, pure fp16, single-fold pipeline validation.
- **v0.3 (Kaggle TPU v5e-8):** Gemma-2-9b-it, pure bf16 (no quantization; 128 GB TPU fits 18 GB model), single fold → all 5 folds. Must select TPU v5e-8 manually in Kaggle UI after push (machine_shape is read-only on CLI push).
- **v0.3b (DGX Spark GB10):** Gemma-2-9b-it in bf16, all 5 folds, no quantization.
- **Expected:** ~0.89–0.92 — the score jump that matters.

### Rung 3b — Reward model zero-shot classifier (v0.4)

Reward models are purpose-trained for human preference prediction — the same task as this
competition. Use one as a **zero-shot or near-zero-shot classifier** by scoring each
response independently and converting the scalar delta to class probabilities.

**Approach:**
1. Load `Skywork-Reward-Llama-3.1-8B-v0.2` (available on Kaggle Models:
   `quincyqiang/skywork-reward-llama-3.1-8b-v0.2`, ~15 GB — fits on T4).
2. Score each (prompt, response_a) and (prompt, response_b) pair → `score_a`, `score_b`.
3. Convert to 3-class probabilities:
   - `p_a = sigmoid((score_a - score_b) * temperature)`
   - `p_b = sigmoid((score_b - score_a) * temperature)`
   - `p_tie`: synthesize from score proximity (e.g. `1 - |p_a - p_b|`), then renormalize
4. Calibrate tie probability on OOF fold; temperature-tune the sigmoid on OOF.

**Variants:**
- **v0.4a** (Kaggle T4, zero-shot): `Skywork-Reward-Llama-3.1-8B-v0.2` — no fine-tuning.
- **v0.4b** (DGX, zero-shot): `Skywork-Reward-Gemma-2-27B` — higher-quality scores, tie
  calibration same approach.
- **v0.4c** (optional): add lightweight classification head on top of frozen reward backbone;
  fine-tune head only on OOF fold to learn tie boundary.

**Why this is interesting:**
- No SFT training required — pure inference → fast iteration
- Reward models see (prompt, response) pairs with the correct task framing
- The 8B version runs on a single T4; the 27B on DGX with no quantization

**Expected:** ~0.90–0.95 zero-shot (reward models have strong priors but tie calibration
is the gap). May match or exceed v0.3 SFT with good calibration.

### Rung 4 — Knowledge distillation (< 0.82 target)
Verified from [cite:3]: the winning approach used:
- **Teacher:** Llama3-70B **or** Qwen2-72B — batch-inference over the full `train.csv`
  in a 5-fold setup, outputting soft `[p_A, p_B, p_tie]` per row.
- **Student:** Gemma2-9B per fold, fine-tuned with LoRA using **KL-divergence** loss
  against teacher's soft targets — not cross-entropy against hard labels.
- **Ensemble:** average the LoRA *layers* (not predictions) from all 5 folds into one
  final model before submission.
- **Why it works:** soft labels carry calibrated confidence (0.75/0.15/0.10 vs
  0.38/0.35/0.27) rather than binary winner flags — the student inherits the teacher's
  understanding of preference margins, not just who won.
- **Hardware:** winners used 8×A100 80GB. DGX Spark GB10 is a single-socket machine —
  Qwen2-72B teacher inference may require quantization (4-bit) or offloading. Qwen2-72B
  → Gemma2-9B is the proven winning recipe; Qwen-32B → Qwen-8B (our existing
  mineral-hr-llm harness) is the practical alternative.
- **Expected:** < 0.82.

#### DGX Spark hardware constraints (120 GB VRAM)

70B models in bf16 (~140 GB) exceed the 120 GB limit (confirmed: Nemotron-3-Nano-30B-A3B
needed ~130 GB with swap just to load weights). Practical model options per role:

| Role | Model | VRAM (bf16) | Notes |
|---|---|---|---|
| Student (5-fold) | `google/gemma-2-9b-it` | ~18 GB | Fits easily; proven winner backbone |
| Student (larger) | `google/gemma-2-27b-it` | ~54 GB | Stronger student if runtime allows |
| Teacher | `Skywork/Skywork-Reward-Gemma-2-27B` | ~54 GB | **Cleanest path** — purpose-built preference reward model; fits in 120 GB with headroom |
| Teacher (alt) | `google/gemma-2-27b-it` | ~54 GB | General instruction model; weaker preference signal than Skywork |
| Teacher (alt) | `Qwen/Qwen2.5-72B` in 4-bit | ~36 GB | Fits quantized; soft-label quality degraded vs bf16 |

**Recommended DGX path (cleanest):**
1. Load `Skywork-Reward-Gemma-2-27B` → batch-infer teacher soft labels over `train.csv`; save to disk.
2. Unload teacher.
3. Load `gemma-2-9b-it` → fine-tune with LoRA + KL-divergence loss against saved soft labels (5 folds).
4. Average LoRA layers across folds → upload adapter to Kaggle for offline inference.

Teacher and student never load simultaneously, so 120 GB is sufficient for each step.

### Rung 5 — Squeeze
- Ensemble Rung 3 (SFT folds) + Rung 4 (distilled folds) weighted average.
- Optionally add DeBERTa-v3-large fold-ensemble for cross-encoder diversity.
- Probability calibration, especially the tie class.

## Cross-validation

- 5-fold stratified on the target. Optionally group/stratify by `model_a`/`model_b`
  pairing for analysis, but never feed those identities to the model.
- Trust CV → leaderboard correlation; the public LB is a small slice.

## Bias handling (this is the competition)

- **Position bias** (judges favor response A): A/B swap augmentation in training +
  swap-TTA at inference.[cite:1]
- **Verbosity bias** (longer ≈ preferred): include length features in baselines; let
  the LLM see full responses but keep the informative **tail** when truncating.[cite:1]
- **Self-enhancement bias:** noted in the brief; not directly actionable without model identities.[cite:1]

## Truncation

Conversations are long. With a fixed token budget, prefer keeping the **end** of each
response (where quality/error signals concentrate) over head truncation.

## Offline submission (do this early)

1. Upload base model (`gemma-2-9b-it`) as a Kaggle **model** or dataset input.
2. Upload trained LoRA adapter as a private Kaggle dataset.
3. Upload any pip wheels needed offline; install with `--no-deps --no-index`.
4. Notebook: `enable_internet=false`, load base + adapter from `/kaggle/input/...`,
   batch-infer ~25k rows within 9 h, write `submission.csv`.

## Runtime budget

Inference on ~25k rows with a 9B model is the binding constraint, not training.
Plan for 4-bit inference and/or vLLM, large batch, max-length cap. Measure wall-clock
on the example test set before the full run.
