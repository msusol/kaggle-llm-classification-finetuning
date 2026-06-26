# Notebook version log

Running record of every Kaggle kernel version: what it tests, errors encountered,
and the outcome. One `##` section per notebook slug.

---

## v0.1-tfidf-baseline

**Kernel id:** `gdataranger/llm-classification-finetuning-v01-tfidf`
**Notebook file:** `notebook/v0.1-tfidf-baseline.ipynb`
**GPU:** none (CPU only) | **Internet:** off

### Context

Phase 1 plumbing baseline: TF-IDF word n-grams (50k features) + 7 length features →
LightGBM 5-fold multiclass. Goal is to lock the submission pipeline and beat the uniform
log loss of 1.0986.

### Investigation Checklist

- [x] Notebook runs end-to-end on Kaggle (no import errors)
- [x] Competition data mounts correctly under `/kaggle/input/`
- [x] `submission.csv` written with correct header and row count
- [x] LB score recorded in leaderboard.md — **1.04157**

### Findings

**v1 (first push):** `FileNotFoundError` on `train.csv` — competition rules had not been
accepted under the `gdataranger` Kaggle account (rules were accepted under a different
account). Data never mounted.

**Root cause confirmed:** Kaggle only mounts `competition_sources` data when the API
account has accepted the competition rules. The `kernel-metadata.json` was also using the
wrong kernel id (`marksusol/...` instead of `gdataranger/...`).

**v3 fix:** Added glob-based path discovery with fallback so the notebook prints the
actual mount path rather than silently failing:

```python
INPUT = next((p for p in _candidates if (p / 'train.csv').exists()), None)
if INPUT is None:
    _found = glob.glob('/kaggle/input/**/train.csv', recursive=True)
    if _found:
        INPUT = Path(_found[0]).parent
```

Fixed `kernel-metadata.json` id → `gdataranger/llm-classification-finetuning-v01-tfidf`.
User accepted rules under `gdataranger`. v3 run succeeded.

### Actions Taken

- Fixed `kernel-metadata.json` kernel id from `marksusol/` to `gdataranger/`
- Added path discovery cell with glob fallback
- User accepted competition rules under `gdataranger` Kaggle account
- Pushed v3; run succeeded

### Resolution

`resolved` — notebook runs cleanly; LB score 1.04157 confirmed (CV 1.0404 ≈ LB,
near-perfect correlation, pipeline validated end-to-end).

### Follow-ups

- Record LB score in `leaderboard.md` once submission is evaluated
- Note: kernel was originally named `llm-classification-finetuning-submission`; renamed to
  `llm-classification-finetuning-v01-tfidf` on 2026-06-26 to match versioned naming convention

---

## v0.2-llama-qlora

**Kernel id:** `gdataranger/llm-classification-finetuning-v02-qlora`
**Notebook file:** `notebook/v0.2-llama-qlora.ipynb`
**GPU:** T4×2 | **Internet:** on (training run) / off (scoring run)

### Context

Phase 3: Llama-3.1-8B LoRA fine-tuned on Kaggle T4×2 as the first end-to-end validation
run. Training and inference in the same notebook. See
[v0.2-gemma-qlora-plan.md](../plans/v0.2-gemma-qlora-plan.md) for full strategy.

### Investigation Checklist

- [ ] Training notebook created
- [ ] LoRA training completes within GPU quota on T4×2
- [ ] OOF log loss recorded
- [ ] Inference < 9 h on test set
- [ ] `submission.csv` passes submission checklist
- [ ] LB score recorded in leaderboard.md
