#!/usr/bin/env bash
set -euo pipefail

SYNTH_CONFIG="${1:-configs/stage1_yourmt3_synth_data_pad05.yaml}"
FINETUNE_CONFIG="${2:-configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml}"
# Always derive the checkout root from this script. This avoids stale absolute
# A2S_PROJECT_DIR values after the project is copied to another cluster path.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs/yourmt3_synth_prepare_logs data/reports
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/yourmt3_synth_prepare_logs/task_${STAMP}.log"

{
  echo "A2S project: $PROJECT_DIR"
  echo "Synthetic render config: $SYNTH_CONFIG"
  echo "YourMT3 finetune config: $FINETUNE_CONFIG"
  echo "Started: $(date -Is)"
  echo "Host: $(hostname)"

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

  export YOURMT3_SOURCE_DIR="${YOURMT3_SOURCE_DIR:-third_party/YourMT3}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

  echo "Python: $(command -v python)"
  python --version

  if ! command -v fluidsynth >/dev/null 2>&1; then
    echo "fluidsynth not found. Install it first, for example: conda install -y -c conda-forge fluidsynth ffmpeg" >&2
    exit 3
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found. Install it first, for example: conda install -y -c conda-forge ffmpeg" >&2
    exit 3
  fi

  python hpc/stage1/check_yourmt3_runtime.py \
    --source_dir "$YOURMT3_SOURCE_DIR" \
    --import_model_helper

  RENDER_ARGS=(
    --config "$SYNTH_CONFIG"
    --splits "${A2S_SYNTH_SPLITS:-train,valid}"
    --jobs "${A2S_SYNTH_JOBS:-8}"
    --skip_existing
  )
  PREPARE_ARGS=(
    --config "$FINETUNE_CONFIG"
    --splits "${A2S_SYNTH_SPLITS:-train,valid}"
    --jobs "${A2S_PREPARE_JOBS:-8}"
    --skip_existing
  )
  if [[ -n "${A2S_SYNTH_LIMIT:-}" ]]; then
    RENDER_ARGS+=(--limit "$A2S_SYNTH_LIMIT")
    PREPARE_ARGS+=(--limit "$A2S_SYNTH_LIMIT")
  fi

  python scripts/render_synthetic_quartets.py "${RENDER_ARGS[@]}"
  python scripts/prepare_yourmt3_quartets.py "${PREPARE_ARGS[@]}"

  python hpc/stage1/check_yourmt3_finetune_data.py \
    --config "$FINETUNE_CONFIG" \
    --full

  echo "Finished: $(date -Is)"
} 2>&1 | tee "$LOG"
