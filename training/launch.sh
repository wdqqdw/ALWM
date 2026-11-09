#!/usr/bin/env bash
set -euo pipefail
stage="${1:?Usage: bash training/launch.sh compressor|p1|p2|p3 [trainer arguments]}"
shift
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python}"
base_model="${BASE_MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
data_root="${DATA_ROOT:-data/ALWM_data}"
manifests="${MANIFEST_DIR:-data/manifests}"
output_root="${OUTPUT_ROOT:-outputs}"
nproc="${NPROC_PER_NODE:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launch=("$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$nproc")
case "$stage" in
  compressor)
    command=("${launch[@]}" training/train_compressor.py
      --model "$base_model" --train-manifest "$manifests/D1.jsonl"
      --video-root "$data_root"
      --output-dir "$output_root/compressor" --compressor c1 --latent-tokens 8
      --bottleneck-dim 1024 --epochs 1 --learning-rate 2e-4 --weight-decay 0.01
      --gradient-accumulation 8 --micro-batch-size 2 --max-grad-norm 1
      --fps 1 --max-frames 64 --video-prefetch-workers 2 --video-prefetch-depth 4
      --duration-bucket-size 1 --attention-backend sdpa
      --checkpoint-every 50 --keep-last-checkpoints 1 --seed 20260822)
    ;;
  p1|p2|p3)
    total=10375
    resume=(--resume-from "")
    case "$stage" in
      p1) stop=2500 ;;
      p2) stop=8500; resume=(--resume-from "$output_root/p1/checkpoints/step-002500") ;;
      p3) stop=10304; total=10304; resume=(--resume-from "$output_root/p2/checkpoints/step-008500") ;;
    esac
    command=("${launch[@]}" training/train_curriculum.py
      --model "$base_model" --ab-train-manifest "$manifests/D2.jsonl"
      --c-train-manifest "$manifests/D3.jsonl" --video-root "$data_root"
      --compressor-checkpoint "$output_root/compressor" --output-dir "$output_root/$stage"
      --total-steps "$total" --a-end-step 2500 --b-end-step 8500 --stop-after-step "$stop"
      --c-b-rehearsal-ratio 0.20 --c-local-data-indexing --learning-rate 1e-5
      --c-learning-rate 2e-6 --c-warmup-ratio 0.05 --warmup-ratio 0.03
      --weight-decay 0.01 --max-grad-norm 1 --alignment-weight 1 --caption-weight 1
      --post-text-weight 1
      --fps 1 --max-frames 16
      --video-prefetch-workers 1 --video-prefetch-depth 8 --duration-bucketing
      --attention-backend sdpa --gradient-checkpointing
      --optimizer adamw8bit --seed 20260823 --log-every 5 --checkpoint-every 0
      "${resume[@]}")
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}" "$@"
  printf '\n'
  exit 0
fi
if [[ -e "$output_root/$stage" ]]; then
  echo "Output already exists: $output_root/$stage. Use a new OUTPUT_ROOT; nothing was overwritten." >&2
  exit 2
fi
[[ -s "$manifests/audit.json" ]] || { echo "Run training/prepare_data.sh first." >&2; exit 2; }
"$python_bin" - "$manifests/audit.json" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1]))
if audit["smoke_subset"] or audit["counts"] != {"D1": 42673, "D2": 15290, "D3": 2885}:
    raise SystemExit("Full training requires full manifests, not a smoke-test subset.")
PY
exec "${command[@]}" "$@"
