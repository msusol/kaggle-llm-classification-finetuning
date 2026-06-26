#!/usr/bin/env zsh
# Download competition data. Requires ~/.kaggle/kaggle.json and accepted rules.
set -euo pipefail

COMP="llm-classification-finetuning"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DEST"

kaggle competitions download -c "$COMP" -p "$DEST"
unzip -o "$DEST/${COMP}.zip" -d "$DEST"
rm -f "$DEST/${COMP}.zip"

echo "Downloaded to $DEST:"
ls -lh "$DEST"
