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

### Rung 3 — Gemma-2-9b + QLoRA (the contender)
- `Gemma2ForSequenceClassification`, 3 labels, 4-bit NF4 + LoRA (r=16–64), bf16 compute.
- Train on DGX Spark; see [v0.2-gemma-qlora-plan.md](v0.2-gemma-qlora-plan.md).
- **Expected:** ~0.89–0.92 — the score jump that matters.

### Rung 4 — Squeeze
- **Swap-TTA:** infer with (a,b) and (b,a), swap the two model probs, average → cancels position bias.
- Ensemble DeBERTa + Gemma; optionally distill a larger teacher.
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
