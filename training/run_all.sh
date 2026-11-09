#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for stage in compressor p1 p2 p3; do
  bash "$script_dir/launch.sh" "$stage"
done
