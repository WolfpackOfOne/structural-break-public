#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv-gpu-tabular}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION"
python -m pip install \
  "numpy==2.4.6" \
  "scipy==1.17.1" \
  "pandas==3.0.5" \
  "scikit-learn==1.9.0" \
  "pyarrow==25.0.1" \
  "psutil==7.2.2" \
  "tabm==0.0.3" \
  "rtdl_num_embeddings==0.0.12" \
  "pytabkit==1.7.3"
python -m pip install -e ".[research]"

python - <<'PY'
import importlib.metadata as md
import torch

required = [
    "torch",
    "tabm",
    "pytabkit",
    "rtdl_num_embeddings",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
]
for pkg in required:
    print(f"{pkg}={md.version(pkg)}")

print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("GPU REQUIRED: CUDA is unavailable")
print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
PY
