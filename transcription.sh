#!/usr/bin/env bash
set -u

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash transcription.sh INPUT_AUDIO_OR_DIR OUTPUT_KERN_DIR [METADATA_FILE_OR_DIR]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_AUDIO_DIR="$1"
OUTPUT_KERN_DIR="$2"
METADATA_DIR="${3:-}"
mkdir -p "$OUTPUT_KERN_DIR/logs"
PYTHON_BIN="${A2S_PYTHON:-python}"
CONFIG_PATH="${A2S_SUBMISSION_CONFIG:-$PROJECT_DIR/configs/pipeline_submission.yaml}"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/ensure_model_assets.py" \
  --config "$CONFIG_PATH" \
  --out_json "$OUTPUT_KERN_DIR/logs/assets.json" \
  >>"$OUTPUT_KERN_DIR/logs/launcher.log" 2>&1
ASSET_STATUS=$?
if [ "$ASSET_STATUS" -ne 0 ]; then
  echo "Model asset preparation failed; see logs/assets.json and logs/launcher.log" >&2
  exit "$ASSET_STATUS"
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/check_submission_assets.py" \
  --config "$CONFIG_PATH" \
  --out_json "$OUTPUT_KERN_DIR/logs/preflight.json" \
  >>"$OUTPUT_KERN_DIR/logs/launcher.log" 2>&1
PREFLIGHT_STATUS=$?
if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
  echo "Submission preflight failed; see logs/preflight.json and logs/launcher.log" >&2
  exit "$PREFLIGHT_STATUS"
fi

ARGS=(
  "$PROJECT_DIR/scripts/run_submission_pipeline.py"
  --config "$CONFIG_PATH"
  --input_audio_dir "$INPUT_AUDIO_DIR"
  --output_kern_dir "$OUTPUT_KERN_DIR"
)
if [ -n "$METADATA_DIR" ]; then
  ARGS+=(--metadata_dir "$METADATA_DIR")
fi
if [ -n "${A2S_LIMIT:-}" ]; then
  ARGS+=(--limit "$A2S_LIMIT")
fi

"$PYTHON_BIN" "${ARGS[@]}" >>"$OUTPUT_KERN_DIR/logs/launcher.log" 2>&1
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "Pipeline exited with status $STATUS; see logs/transcription.log and logs/launcher.log" >&2
fi
exit "$STATUS"
