#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-valid}"
LIMIT="${2:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="${A2S_STAGE2_MIDI_CONFIG:-configs/stage2_midi_pred_events_long_v2_infer.yaml}"
EVENTS_DIR="${A2S_STAGE2_EVENTS_DIR:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_${SPLIT}_voice_model/events}"
OUT_ROOT="${A2S_STAGE2_OUT_ROOT:-outputs/stage2_midi_pred_events_long_v2_from_pad05_nops_voice_model_${SPLIT}}"
DEFAULT_MANIFEST="data/manifests_dedup/${SPLIT}.csv"
if [[ ! -f "$DEFAULT_MANIFEST" ]]; then
  DEFAULT_MANIFEST="data/manifests/${SPLIT}.csv"
fi
MANIFEST="${A2S_STAGE2_MANIFEST:-$DEFAULT_MANIFEST}"

mkdir -p "$OUT_ROOT/metrics"

echo "A2S Stage2 MIDI pred-events long-v2 inference"
echo "Project: $PROJECT_DIR"
echo "Split: $SPLIT"
echo "Config: $CONFIG"
echo "Events: $EVENTS_DIR"
echo "Output: $OUT_ROOT"

LIMIT_ARGS=()
PRED_ONLY_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
  PRED_ONLY_ARGS=(--pred_only)
fi
if [[ "${A2S_STAGE2_PRED_ONLY:-0}" == "1" ]]; then
  PRED_ONLY_ARGS=(--pred_only)
fi

python scripts/infer_stage2_midi.py \
  --config "$CONFIG" \
  --events_dir "$EVENTS_DIR" \
  --manifest "$MANIFEST" \
  --out_dir "$OUT_ROOT" \
  "${LIMIT_ARGS[@]}"

python scripts/validate_kern.py \
  --config "$CONFIG" \
  --input_dir "$OUT_ROOT/kern" \
  --out_json "$OUT_ROOT/metrics/validation.json" || true

python scripts/evaluate_stage2.py \
  --config "$CONFIG" \
  --pred_dir "$OUT_ROOT/kern" \
  --manifest "$MANIFEST" \
  --out_json "$OUT_ROOT/metrics/stage2_metrics.json" \
  "${LIMIT_ARGS[@]}" \
  "${PRED_ONLY_ARGS[@]}"

echo "Stage2 MIDI pred-events long-v2 metrics: $OUT_ROOT/metrics/stage2_metrics.json"
