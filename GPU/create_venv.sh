#!/usr/bin/env bash
set -euo pipefail

# GPU/create_venv.sh
# Create a venv inside the GPU folder and install requirements.
# Usage (from repo root or GPU folder):
#  ./GPU/create_venv.sh [venv-dir] [python-executable]
# Examples:
#  ./GPU/create_venv.sh           -> creates GPU/.venv using python3
#  ./GPU/create_venv.sh venv_gpu  -> creates GPU/venv_gpu using python3
#  ./GPU/create_venv.sh .venv python3.11 -> use specific python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR_REL="${1:-.venv}"
PYTHON="${2:-python3}"

VENV_PATH="$SCRIPT_DIR/$VENV_DIR_REL"

echo "Using python interpreter: $PYTHON"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: '$PYTHON' not found in PATH." >&2
  exit 1
fi

if [ -d "$VENV_PATH" ]; then
  echo "Virtualenv directory '$VENV_PATH' already exists. Skipping creation."
else
  echo "Creating virtualenv at '$VENV_PATH'..."
  "$PYTHON" -m venv "$VENV_PATH"
  echo "Virtualenv created."
fi

echo "Upgrading pip, setuptools, wheel inside venv..."
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel

# Prefer GPU/requirements.txt, fall back to repo-level requirements.txt
REQ_FILE=""
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  REQ_FILE="$SCRIPT_DIR/requirements.txt"
elif [ -f "$SCRIPT_DIR/../requirements.txt" ]; then
  REQ_FILE="$SCRIPT_DIR/../requirements.txt"
fi

if [ -n "$REQ_FILE" ]; then
  echo "Installing packages from $REQ_FILE..."
  "$VENV_PATH/bin/pip" install -r "$REQ_FILE"
  echo "Requirements installed."
else
  echo "No requirements.txt found in GPU or repo root — nothing to install."
fi

echo
echo "Done. To start using the GPU virtualenv run:"
echo "  source \"$VENV_PATH/bin/activate\""
echo "Or run commands with the venv Python:\n  $VENV_PATH/bin/python path/to/script.py"
