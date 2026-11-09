#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" inference/run.py --model "${MODEL_DIR:-models/ALWM}" --data-root "${DATA_ROOT:-data/ALWM_data}" "$@"
