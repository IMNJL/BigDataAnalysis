#!/usr/bin/env bash
# setup_env.sh - create environment for the Autolabelling Snorkel project
#
# Usage:
#   ./setup_env.sh conda    # create a conda env (recommended on macOS)
#   ./setup_env.sh venv     # create a Python venv using Python 3.11 (pyenv recommended)
#
set -euo pipefail

ACTION=${1:-}
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

print_header(){
  echo
  echo "==== $1 ===="
}

if [[ "$ACTION" == "conda" ]]; then
  print_header "Creating conda environment (recommended)"
  # Use conda-forge for prebuilt macOS/arm64 binaries
  ENV_NAME=${2:-bda_snorkel}
  echo "Creating conda env '$ENV_NAME' with Python 3.11 and commonly used packages..."
  conda create -n "$ENV_NAME" python=3.11 -y -c conda-forge \
    scikit-learn=1.3.2 snorkel=0.10.0 pandas=2.2.3 numpy=1.26.4 tqdm spacy pip
  echo "To activate: conda activate $ENV_NAME"
  echo "Install label-studio and optionally torch in the activated env (examples):"
  echo "  pip install label-studio"
  echo "  # For CPU-only PyTorch on macOS: follow https://pytorch.org and pick the appropriate wheel"
  exit 0
fi

if [[ "$ACTION" == "venv" ]]; then
  print_header "Creating Python venv (pyenv recommended)"
  # Prefer pyenv-installed Python 3.11
  # Choose Python command: argument or try common candidates
  REQUESTED_PY=${2:-}
  if [[ -n "$REQUESTED_PY" ]]; then
    PYTHON_CMD=$REQUESTED_PY
  else
    PYTHON_CMD=""
    for cand in python3.11 python3 python; do
      if command -v "$cand" >/dev/null 2>&1; then
        PYTHON_CMD=$cand
        break
      fi
    done
  fi

  if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo "Error: Python command '$PYTHON_CMD' not found."
    echo "Please install Python 3.11 (pyenv or system package) or pass a python executable as the second argument, e.g.:"
    echo "  ./setup_env.sh venv /path/to/python3.11"
    exit 1
  fi

  echo "Using python command: $PYTHON_CMD"

  # Check Python version
  VER=$($PYTHON_CMD -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')
  echo "Python version: $VER"
  echo "Creating venv at .venv and activating..."
  $PYTHON_CMD -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel cython

  echo "Important: On macOS, building scikit-learn may require Xcode CLT and libomp."
  echo "If you get OpenMP errors, run:"
  echo "  xcode-select --install"
  echo "  brew install libomp"

  echo "Installing core packages via pip (this will try to use wheels if available):"
  pip install scikit-learn==1.3.2 snorkel==0.10.0 pandas==2.2.3 numpy==1.26.4 tqdm
  echo "If scikit-learn fails to build, prefer the conda path or use Python 3.11 wheels."
  echo "Install the rest of requirements (label-studio, transformers, torch) as needed:"
  echo "  pip install -r $PROJECT_DIR/requirements.txt"
  exit 0
fi

cat <<'EOF'
Usage: setup_env.sh [conda|venv]

Examples:
  # Recommended (conda-forge) on macOS/Apple Silicon
  ./setup_env.sh conda myenv

  # venv using system or pyenv python3 (must be Python 3.11)
  ./setup_env.sh venv python3.11

Notes:
- If you use macOS arm64 (M1/M2) install Miniforge/Miniconda/Conda-forge to get prebuilt wheels.
- Building scikit-learn from source on macOS often fails unless you have the right compiler toolchain and libomp.
EOF

exit 1
