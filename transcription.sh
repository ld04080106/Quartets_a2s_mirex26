#!/usr/bin/env bash
set -u

if [ "$#" -ne 3 ]; then
  echo "Error: this submission supports Staves-Informed A2S only; metadata is required." >&2
  echo "Usage: bash transcription.sh INPUT_AUDIO_OR_DIR OUTPUT_KERN_DIR METADATA_FILE_OR_DIR" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_AUDIO_DIR="$1"
OUTPUT_KERN_DIR="$2"
METADATA_DIR="$3"
if [ ! -e "$METADATA_DIR" ]; then
  echo "Error: metadata path does not exist: $METADATA_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_KERN_DIR/logs"
PYTHON_BIN="${A2S_PYTHON:-python}"
CONFIG_PATH="${A2S_SUBMISSION_CONFIG:-$PROJECT_DIR/configs/pipeline_submission.yaml}"

VALIDATION_ARGS=(
  "$PROJECT_DIR/scripts/run_submission_pipeline.py"
  --config "$CONFIG_PATH"
  --input_audio_dir "$INPUT_AUDIO_DIR"
  --output_kern_dir "$OUTPUT_KERN_DIR"
  --metadata_dir "$METADATA_DIR"
  --validate_inputs_only
)
if [ -n "${A2S_LIMIT:-}" ]; then
  VALIDATION_ARGS+=(--limit "$A2S_LIMIT")
fi

echo "[1/4] Validating Staves-Informed inputs ..."
"$PYTHON_BIN" "${VALIDATION_ARGS[@]}"
INPUT_STATUS=$?
if [ "$INPUT_STATUS" -ne 0 ]; then
  echo "Input validation failed; Staves-Informed metadata is required for every audio file." >&2
  exit "$INPUT_STATUS"
fi

echo "[2/4] Checking required model assets ..."
"$PYTHON_BIN" "$PROJECT_DIR/scripts/ensure_model_assets.py" \
  --config "$CONFIG_PATH" \
  --out_json "$OUTPUT_KERN_DIR/logs/assets.json"
ASSET_STATUS=$?
if [ "$ASSET_STATUS" -ne 0 ]; then
  echo "Model asset preparation failed; see logs/assets.json and logs/launcher.log" >&2
  exit "$ASSET_STATUS"
fi

echo "[3/4] Running submission preflight ..."
"$PYTHON_BIN" "$PROJECT_DIR/scripts/check_submission_assets.py" \
  --config "$CONFIG_PATH" \
  --out_json "$OUTPUT_KERN_DIR/logs/preflight.json" \
  >>"$OUTPUT_KERN_DIR/logs/launcher.log" 2>&1
PREFLIGHT_STATUS=$?
if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
  echo "Submission preflight failed; see logs/preflight.json and logs/launcher.log" >&2
  exit "$PREFLIGHT_STATUS"
fi
echo "Preflight passed."

ARGS=(
  "$PROJECT_DIR/scripts/run_submission_pipeline.py"
  --config "$CONFIG_PATH"
  --input_audio_dir "$INPUT_AUDIO_DIR"
  --output_kern_dir "$OUTPUT_KERN_DIR"
)
ARGS+=(--metadata_dir "$METADATA_DIR")
if [ -n "${A2S_LIMIT:-}" ]; then
  ARGS+=(--limit "$A2S_LIMIT")
fi

echo "[4/4] Transcribing input audio ..."
"$PYTHON_BIN" "${ARGS[@]}" 2>&1 | tee -a "$OUTPUT_KERN_DIR/logs/launcher.log"
STATUS=${PIPESTATUS[0]}
if [ "$STATUS" -ne 0 ]; then
  echo "Pipeline exited with status $STATUS; see logs/transcription.log and logs/launcher.log" >&2
fi
exit "$STATUS"
