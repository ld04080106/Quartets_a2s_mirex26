#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:?usage: run_yourmt3_synth_finetuned_nops_eval.sh SPLIT [LIMIT]}"
LIMIT="${2:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${A2S_CONFIG:-$PROJECT_DIR/configs/stage1_yourmt3_synth_finetuned_pad05_nops_hpc.yaml}"
OUTPUT_ROOT="${A2S_OUTPUT_ROOT:-$PROJECT_DIR/outputs/stage1_yourmt3_synth_finetuned_pad05_nops_${SPLIT}}"
DEFAULT_MANIFEST="$PROJECT_DIR/data/manifests_dedup/$SPLIT.csv"
if [[ ! -f "$DEFAULT_MANIFEST" ]]; then
  DEFAULT_MANIFEST="$PROJECT_DIR/data/manifests/$SPLIT.csv"
fi
MANIFEST="${A2S_MANIFEST:-$DEFAULT_MANIFEST}"
ORACLE_DIR="${A2S_ORACLE_DIR:-$PROJECT_DIR/data/oracle_events/$SPLIT}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
else
    echo "Cannot find conda. Put it on PATH, source it first, or set CONDA_SH to conda.sh." >&2
    exit 2
fi

if [[ -n "${YOURMT3_ENV_PREFIX:-}" ]]; then
    conda activate "$YOURMT3_ENV_PREFIX"
else
    conda activate yourmt3
fi

cd "$PROJECT_DIR"
export YOURMT3_SOURCE_DIR="${YOURMT3_SOURCE_DIR:-hpc_assets/stage1/sources/YourMT3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUTPUT_ROOT/events" "$OUTPUT_ROOT/midi" "$OUTPUT_ROOT/metrics"

echo "Using YourMT3 source: $YOURMT3_SOURCE_DIR"
echo "Config: $CONFIG"
echo "Manifest: $MANIFEST"
echo "Output root: $OUTPUT_ROOT"
echo "Oracle dir: $ORACLE_DIR"
python - "$CONFIG" "$YOURMT3_SOURCE_DIR" <<'PY'
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML unavailable; cannot print checkpoint summary")
    raise SystemExit(0)

config_path = Path(sys.argv[1])
source_dir = Path(sys.argv[2])
payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
hpc = payload.get("hpc_stage1", {})
checkpoint = hpc.get("checkpoint_path")
if checkpoint:
    checkpoint_path = Path(os.path.expandvars(str(checkpoint)))
    if not checkpoint_path.is_absolute():
        checkpoint_path = source_dir / checkpoint_path
else:
    checkpoint_path = None
print("Model name:", hpc.get("model_name"))
print("Configured checkpoint:", checkpoint_path)
PY

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 2
fi
if [[ "$(basename "$MANIFEST")" != "$SPLIT.csv" && "${A2S_ALLOW_MANIFEST_MISMATCH:-0}" != "1" ]]; then
  echo "Manifest/split mismatch: split=$SPLIT manifest=$MANIFEST" >&2
  echo "Unset stale A2S_MANIFEST or set A2S_ALLOW_MANIFEST_MISMATCH=1 intentionally." >&2
  exit 2
fi

python - "$MANIFEST" "$SPLIT" "$OUTPUT_ROOT/events" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
split = sys.argv[2]
events_dir = Path(sys.argv[3])
with manifest.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"empty manifest: {manifest}")
first = rows[0].get("sample_id", "")
if not first.startswith(f"{split}_") and not first.startswith(f"{split[:3]}_") and split != "valid":
    raise SystemExit(f"manifest first sample_id {first!r} does not look like split {split!r}")
if split == "valid" and not (first.startswith("valid_") or first.startswith("val_")):
    raise SystemExit(f"manifest first sample_id {first!r} does not look like valid split")
if events_dir.exists():
    stale = sorted(
        path.name for path in events_dir.glob("*.json")
        if path.name != "stage1_infer_report.json"
        and not path.name.startswith((f"{split}_", "val_" if split == "valid" else f"{split}_"))
    )
    if stale and not bool(int(__import__("os").environ.get("A2S_ALLOW_MIXED_OUTPUT_PREFIX", "0"))):
        raise SystemExit(
            "output events dir contains files from another split, e.g. "
            + ", ".join(stale[:5])
            + "\nUse a clean A2S_OUTPUT_ROOT or set A2S_ALLOW_MIXED_OUTPUT_PREFIX=1 intentionally."
        )
print(f"Manifest check ok: {manifest} first_sample_id={first} rows={len(rows)}")
PY

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then LIMIT_ARGS=(--limit "$LIMIT"); fi

python hpc/stage1/infer_manifest.py \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --out_dir "$OUTPUT_ROOT/events" --midi_dir "$OUTPUT_ROOT/midi" \
  --skip_existing "${LIMIT_ARGS[@]}"

python scripts/evaluate_stage1.py \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --pred_dir "$OUTPUT_ROOT/events" --oracle_dir "$ORACLE_DIR" \
  --out_json "$OUTPUT_ROOT/metrics/stage1_metrics.json" "${LIMIT_ARGS[@]}"

echo "Nops synthetic fine-tuned Stage 1 metrics: $OUTPUT_ROOT/metrics/stage1_metrics.json"
