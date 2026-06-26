#!/usr/bin/env zsh
# Usage: zsh scripts/push_notebook.sh <slug>
# Example: zsh scripts/push_notebook.sh v0.1-tfidf-baseline
#
# Stages notebook/<slug>.ipynb + notebook/<slug>-kernel-metadata.json
# into a temp dir (Kaggle CLI requires the file be named kernel-metadata.json)
# and pushes to Kaggle.

set -euo pipefail

SLUG="${1:?Usage: $0 <slug>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOTEBOOK="$REPO_ROOT/notebook/${SLUG}.ipynb"
METADATA="$REPO_ROOT/notebook/${SLUG}-kernel-metadata.json"

[[ -f "$NOTEBOOK" ]] || { echo "ERROR: $NOTEBOOK not found"; exit 1; }
[[ -f "$METADATA" ]] || { echo "ERROR: $METADATA not found"; exit 1; }

STAGE="$(mktemp -d)"
trap "rm -rf '$STAGE'" EXIT

cp "$NOTEBOOK" "$STAGE/${SLUG}.ipynb"
cp "$METADATA" "$STAGE/kernel-metadata.json"

echo "Pushing $SLUG …"
kaggle kernels push -p "$STAGE"
