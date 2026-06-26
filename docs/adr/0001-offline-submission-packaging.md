# 0001 — Offline submission packaging

## Status
Accepted

## Context
This is a Code Competition: the scoring notebook runs with **internet disabled** and a
**9 h** runtime cap, and must emit `submission.csv`.[cite:1] A 9B model plus adapter
cannot be fetched at runtime.

## Decision
Package everything as Kaggle inputs and load offline:
- base model (`gemma-2-9b-it`) as a Kaggle model/dataset input,
- trained LoRA adapter as a private Kaggle dataset,
- any required pip wheels, installed with `--no-deps --no-index`.

The inference notebook sets `enable_internet=false`, loads base + adapter from
`/kaggle/input/...`, runs swap-TTA batched inference within 9 h, and writes
`submission.csv`.

## Consequences
- Every dependency must be pre-staged; no `from_pretrained` network calls.
- Runtime is the binding constraint → 4-bit / batched / vLLM inference.
- Adapter dataset must be re-versioned and re-attached on each retrain.
