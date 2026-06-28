#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

echo "=== 1. Pausing background services ==="
bash "$SCRIPT_DIR/services.sh" pause

trap 'echo "=== Exiting: Resuming background services ===" && bash "$SCRIPT_DIR/services.sh" resume' EXIT

echo "=== 2. Pre-flight GPU Check ==="
used_mem=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [[ -n "$used_mem" && "$used_mem" -gt 1000 ]]; then
    echo "ERROR: GPU is currently in use (${used_mem} MiB). Aborting to prevent OOM."
    exit 1
fi
echo "GPU memory clear (${used_mem:-0} MiB used)."

echo "=== 3. Starting Training (v0.3b 5-fold) ==="
docker run --rm -t \
    --name "gemma-2-9b-trainer" \
    --oom-score-adj 300 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
    -e HF_HOME="/workspace/.cache/huggingface" \
    -v "${WORKSPACE}":/workspace \
    -w /workspace \
    gemma-trainer-gb10 \
    python scripts/train_v03b.py

echo "=== 4. Averaging Adapters ==="
docker run --rm -t \
    --name "gemma-2-9b-averager" \
    --oom-score-adj 300 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
    -e HF_HOME="/workspace/.cache/huggingface" \
    -v "${WORKSPACE}":/workspace \
    -w /workspace \
    gemma-trainer-gb10 \
    python scripts/average_lora.py \
        --base-model "google/gemma-2-2b-it" \
        --adapter-dirs \
            "/workspace/output/v0.3b_adapters/fold_0" \
            "/workspace/output/v0.3b_adapters/fold_1" \
            "/workspace/output/v0.3b_adapters/fold_2" \
            "/workspace/output/v0.3b_adapters/fold_3" \
            "/workspace/output/v0.3b_adapters/fold_4" \
        --out-dir "/workspace/output/v0.3b_averaged_adapter"

echo "=== Pipeline Completed Successfully ==="


