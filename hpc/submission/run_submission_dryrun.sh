#!/usr/bin/env bash
set -euo pipefail

INPUT_AUDIO_DIR="${1:-data/raw/quartets}"
OUTPUT_KERN_DIR="${2:-outputs/mirex_submission_dryrun}"
METADATA_DIR="${3:-}"
LIMIT="${A2S_LIMIT:-${4:-}}"
PYTHON_BIN="${A2S_PYTHON:-python}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_KERN_DIR/logs"

echo "A2S MIREX submission dry-run"
echo "Project: $PROJECT_DIR"
echo "Input audio: $INPUT_AUDIO_DIR"
echo "Output kern: $OUTPUT_KERN_DIR"
echo "Metadata: ${METADATA_DIR:-<none>}"
echo "Limit: ${LIMIT:-<none>}"

if [[ -n "$METADATA_DIR" ]]; then
  A2S_LIMIT="$LIMIT" bash transcription.sh "$INPUT_AUDIO_DIR" "$OUTPUT_KERN_DIR" "$METADATA_DIR"
else
  A2S_LIMIT="$LIMIT" bash transcription.sh "$INPUT_AUDIO_DIR" "$OUTPUT_KERN_DIR"
fi

CHECK_ARGS=(
  scripts/check_submission_outputs.py
  --input_audio "$INPUT_AUDIO_DIR"
  --output_dir "$OUTPUT_KERN_DIR"
  --out_json "$OUTPUT_KERN_DIR/logs/submission_contract.json"
)
if [[ -n "$LIMIT" ]]; then
  CHECK_ARGS+=(--limit "$LIMIT")
fi
"$PYTHON_BIN" "${CHECK_ARGS[@]}"

echo "Dry-run report: $OUTPUT_KERN_DIR/logs/pipeline_report.json"
echo "Contract check: $OUTPUT_KERN_DIR/logs/submission_contract.json"
