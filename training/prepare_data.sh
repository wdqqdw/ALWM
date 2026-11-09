#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
exec "${PYTHON_BIN:-python}" training/prepare_data.py --data-root "${DATA_ROOT:-data/ALWM_data}" --output-dir "${MANIFEST_DIR:-data/manifests}" "$@"
