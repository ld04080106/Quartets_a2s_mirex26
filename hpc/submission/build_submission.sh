#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_BASE="${1:-$PROJECT_DIR/outputs/mirex2026_a2s_submission}"
PYTHON_BIN="${A2S_PYTHON:-python}"
cd "$PROJECT_DIR"

"$PYTHON_BIN" scripts/check_submission_assets.py \
  --config configs/pipeline_submission.yaml
"$PYTHON_BIN" scripts/package_submission.py \
  --config configs/pipeline_submission.yaml \
  --out "$OUTPUT_BASE" \
  --include_runtime_assets \
  --format tar.gz

echo "Archive ready. Unpack it into a clean directory and run an 8-file smoke test before submission."
