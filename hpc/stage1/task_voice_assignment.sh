#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage1_voice_assignment.yaml}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs/voice_assignment/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/voice_assignment/logs/task_${STAMP}.log"

{
  echo "A2S project: $PROJECT_DIR"
  echo "Config: $CONFIG"
  echo "Started: $(date -Is)"

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

  TRAIN_ARGS=(--config "$CONFIG")
  if [[ -n "${A2S_VOICE_TRAIN_PRED_DIR:-}" ]]; then
    TRAIN_ARGS+=(--train_pred_dir "$A2S_VOICE_TRAIN_PRED_DIR")
  fi
  if [[ -n "${A2S_VOICE_ORACLE_TRAIN_DIR:-}" ]]; then
    TRAIN_ARGS+=(--oracle_train_dir "$A2S_VOICE_ORACLE_TRAIN_DIR")
  fi
  python scripts/train_voice_assignment.py "${TRAIN_ARGS[@]}"

  python scripts/infer_voice_assignment.py \
    --config "$CONFIG" \
    --pred_dir "${A2S_VOICE_VALID_PRED_DIR:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_valid/events}" \
    --out_dir "${A2S_VOICE_VALID_OUT_DIR:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_valid_voice_model/events}" \
    --midi_dir "${A2S_VOICE_VALID_MIDI_DIR:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_valid_voice_model/voice_midi}"

  python scripts/evaluate_voice_assignment.py \
    --config "$CONFIG" \
    --pred_dir "${A2S_VOICE_VALID_OUT_DIR:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_valid_voice_model/events}" \
    --oracle_dir "${A2S_VOICE_ORACLE_VALID_DIR:-data/oracle_events/valid}" \
    --out_json "${A2S_VOICE_VALID_METRICS:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_valid_voice_model/metrics/voice_assignment_metrics.json}"

  python scripts/evaluate_stage1.py \
    --config configs/stage1_yourmt3_synth_finetuned_pad05_nops_hpc.yaml \
    --pred_dir "${A2S_VOICE_VALID_OUT_DIR:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_valid_voice_model/events}" \
    --oracle_dir "${A2S_VOICE_ORACLE_VALID_DIR:-data/oracle_events/valid}" \
    --out_json "${A2S_VOICE_VALID_STAGE1_METRICS:-outputs/stage1_yourmt3_synth_finetuned_pad05_nops_valid_voice_model/metrics/stage1_metrics.json}" \
    --pred_only

  echo "Finished: $(date -Is)"
} 2>&1 | tee "$LOG"
