# Submission Checklist

## File format
- [ ] File is named exactly `submission.csv`.[cite:1]
- [ ] Header is `id,winner_model_a,winner_model_b,winner_tie`.[cite:1]
- [ ] One row per test `id`; row count matches test set.
- [ ] Three probabilities per row sum to ~1.0; no NaN / negative values.

## Offline / environment
- [ ] Notebook `enable_internet=false`.[cite:1]
- [ ] Base model + adapter + wheels loaded from `/kaggle/input/...` (no downloads).
- [ ] No `from_pretrained(..., token=...)` or any network call at runtime.

## Runtime
- [ ] Full hidden-test inference verified < 9 h (measured on example test first).[cite:1]
- [ ] Batch size / max-length tuned; OOM-safe.

## Modeling sanity
- [ ] Labels map correctly: 0=winner_model_a, 1=winner_model_b, 2=winner_tie.
- [ ] Swap-TTA averaging swaps the a/b probabilities correctly (tie unchanged).
- [ ] OOF log loss recorded in leaderboard.md and beats current best.

## Competition hygiene
- [ ] `model_a`/`model_b` not used as features anywhere.
- [ ] Rolling-leaderboard: re-submit within 2 months to stay ranked.[cite:1]
