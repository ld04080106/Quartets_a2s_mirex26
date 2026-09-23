#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${A2S_PYTHON:-python}"

"$PYTHON_BIN" -m pip install --upgrade "setuptools<81" wheel
"$PYTHON_BIN" -m pip install -r "$PROJECT_DIR/requirements_submission.txt"

echo "Install a CUDA-compatible torch/torchaudio build separately if they are not already available."
echo "Then run: $PYTHON_BIN $PROJECT_DIR/scripts/ensure_model_assets.py --config $PROJECT_DIR/configs/pipeline_submission.yaml"
echo "Follow with: $PYTHON_BIN $PROJECT_DIR/scripts/check_submission_assets.py --config $PROJECT_DIR/configs/pipeline_submission.yaml"
