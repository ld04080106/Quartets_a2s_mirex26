#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage2_midi_train_pred_events_long_v2.yaml}"
LIMIT="${2:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "A2S Stage2 MIDI pred-events training"
echo "Project: $PROJECT_DIR"
echo "Config: $CONFIG"
echo "Started: $(date -Iseconds)"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

python scripts/prepare_stage2_midi_data.py \
  --config "$CONFIG" \
  "${LIMIT_ARGS[@]}"

python scripts/check_stage2_midi_data.py \
  --config "$CONFIG"

python scripts/train_stage2_midi.py \
  --config "$CONFIG"

echo "Finished: $(date -Iseconds)"
