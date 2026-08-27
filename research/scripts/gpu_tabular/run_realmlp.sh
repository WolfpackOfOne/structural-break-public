#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

if [[ -f ".venv-gpu-tabular/bin/activate" ]]; then
  source ".venv-gpu-tabular/bin/activate"
fi

CONFIG="${CONFIG:-research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json}"
OUTPUT_ROOT="${SBR_GPU_OOF_ROOT:-research/oof/gpu_tabular_2026}"

args=(--config "$CONFIG" --output-root "$OUTPUT_ROOT" --resume)
if [[ -n "${SBR_ARTIFACT_ROOT:-}" ]]; then
  args+=(--artifact-root "$SBR_ARTIFACT_ROOT")
fi

python research/scripts/gpu_tabular/realmlp_runner.py "${args[@]}" "$@"
