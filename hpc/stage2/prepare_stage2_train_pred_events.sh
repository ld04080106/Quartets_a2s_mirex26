#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-train}"
LIMIT="${2:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

STAGE1_CONFIG="${A2S_STAGE1_CONFIG:-configs/stage1_yourmt3_synth_finetuned_pad05_nops_hpc.yaml}"
VOICE_CONFIG="${A2S_VOICE_CONFIG:-configs/stage1_voice_assignment.yaml}"
STAGE1_RAW_ROOT="${A2S_STAGE1_RAW_ROOT:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_${SPLIT}}"
VOICE_OUT_ROOT="${A2S_VOICE_OUT_ROOT:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_${SPLIT}_voice_model}"
VOICE_MODEL="${A2S_VOICE_MODEL:-outputs/voice_assignment/model.pkl}"
DEFAULT_MANIFEST="data/manifests_dedup/${SPLIT}.csv"
if [[ ! -f "$DEFAULT_MANIFEST" ]]; then
  DEFAULT_MANIFEST="data/manifests/${SPLIT}.csv"
fi
MANIFEST="${A2S_MANIFEST:-$DEFAULT_MANIFEST}"
ORACLE_DIR="${A2S_ORACLE_DIR:-data/oracle_events/${SPLIT}}"

mkdir -p "$STAGE1_RAW_ROOT/events" "$STAGE1_RAW_ROOT/midi" "$STAGE1_RAW_ROOT/metrics"
mkdir -p "$VOICE_OUT_ROOT/events" "$VOICE_OUT_ROOT/voice_midi" "$VOICE_OUT_ROOT/metrics"

echo "A2S prepare Stage2 train pred-events"
echo "Project: $PROJECT_DIR"
echo "Split: $SPLIT"
echo "Stage1 config: $STAGE1_CONFIG"
echo "Voice config: $VOICE_CONFIG"
echo "Manifest: $MANIFEST"
echo "Stage1 raw root: $STAGE1_RAW_ROOT"
echo "Voice output root: $VOICE_OUT_ROOT"
echo "Voice model: $VOICE_MODEL"
echo "Started: $(date -Iseconds)"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$VOICE_MODEL" ]]; then
  echo "Missing voice assignment model: $VOICE_MODEL" >&2
  echo "Run hpc/stage1/task_voice_assignment.sh first, or set A2S_VOICE_MODEL." >&2
  exit 2
fi

LIMIT_ARGS=()
PRED_ONLY_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
  PRED_ONLY_ARGS=(--pred_only)
fi

python hpc/stage1/infer_manifest.py \
  --config "$STAGE1_CONFIG" \
  --manifest "$MANIFEST" \
  --out_dir "$STAGE1_RAW_ROOT/events" \
  --midi_dir "$STAGE1_RAW_ROOT/midi" \
  --skip_existing \
  "${LIMIT_ARGS[@]}"

python scripts/evaluate_stage1.py \
  --config "$STAGE1_CONFIG" \
  --manifest "$MANIFEST" \
  --pred_dir "$STAGE1_RAW_ROOT/events" \
  --oracle_dir "$ORACLE_DIR" \
  --out_json "$STAGE1_RAW_ROOT/metrics/stage1_metrics.json" \
  "${LIMIT_ARGS[@]}" \
  "${PRED_ONLY_ARGS[@]}" || true

python scripts/infer_voice_assignment.py \
  --config "$VOICE_CONFIG" \
  --model "$VOICE_MODEL" \
  --pred_dir "$STAGE1_RAW_ROOT/events" \
  --out_dir "$VOICE_OUT_ROOT/events" \
  --midi_dir "$VOICE_OUT_ROOT/voice_midi" \
  --skip_existing \
  "${LIMIT_ARGS[@]}"

python scripts/evaluate_voice_assignment.py \
  --config "$VOICE_CONFIG" \
  --pred_dir "$VOICE_OUT_ROOT/events" \
  --oracle_dir "$ORACLE_DIR" \
  --out_json "$VOICE_OUT_ROOT/metrics/voice_assignment_metrics.json" \
  "${LIMIT_ARGS[@]}" || true

python scripts/evaluate_stage1.py \
  --config "$STAGE1_CONFIG" \
  --manifest "$MANIFEST" \
  --pred_dir "$VOICE_OUT_ROOT/events" \
  --oracle_dir "$ORACLE_DIR" \
  --out_json "$VOICE_OUT_ROOT/metrics/stage1_metrics.json" \
  "${LIMIT_ARGS[@]}" \
  "${PRED_ONLY_ARGS[@]}" || true

python - "$MANIFEST" "$VOICE_OUT_ROOT/events" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
events_dir = Path(sys.argv[2])
with manifest.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
expected = {row["sample_id"] for row in rows}
actual = {path.stem for path in events_dir.glob("*.json") if path.name != "voice_assignment_infer_report.json"}
missing = sorted(expected - actual)
print(f"voice-model events: expected={len(expected)} actual={len(actual)} missing={len(missing)}")
if missing:
    print("missing preview:", ", ".join(missing[:20]))
    raise SystemExit(2)
PY

echo "Finished: $(date -Iseconds)"
echo "Prepared voice-model events: $VOICE_OUT_ROOT/events"
