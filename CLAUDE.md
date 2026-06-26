# CLAUDE.md

Working rules for this competition project.

## Documentation framework

`docs/` is the source of truth. Canonical folder roles:

- `docs/plans/` — planning documents. Versioned plans are named `vX.Y-<slug>-plan.md`.
  Keep `docs/plans/TODO.md` in sync with the active plan.
- `docs/plans/competition-overview.md` — frozen summary of rules, data, metric, deadlines.
- `docs/plans/implementation-plan.md` — the overall strategy ladder.
- `docs/plans/leaderboard.md` — update after **every** completed training run + validation pass.
- `docs/plans/submission-checklist.md` — run through before every Kaggle submission.
- `docs/plans/CITATIONS.md` — register every `[cite:N]` source; N = max existing + 1.
- `docs/adr/` — Architecture Decision Records, `NNNN-title.md`, one decision each.
- `docs/investigate/` — investigation logs (symptom → hypothesis → finding).

## Citations

Use `[cite:N]` inline for any external claim (rules, discussion, paper). Register the
source in `docs/plans/CITATIONS.md`. Never reuse a number.

## Kaggle workflow

- All notebook changes go through `kaggle kernels push`; never instruct UI edits.
- Submission must be **offline**: base model, adapter, and any pip wheels are uploaded
  as Kaggle datasets and installed/loaded from `/kaggle/input/...`.
- Submission file is exactly `submission.csv` with header
  `id,winner_model_a,winner_model_b,winner_tie`.

## DGX Spark training rules

- Train on `sparkdb62` (GB10). Always run long jobs under `tmux`; never background a
  training process in a way that dies on broken pipe.
- Reuse the working LoRA/PEFT stack from `mineral-hr-llm` as the starting harness.

## Commits

Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`). Update `README.md` layout +
commands whenever scripts, data files, or configs change.
