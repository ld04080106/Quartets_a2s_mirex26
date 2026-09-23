#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml}"
# Always derive the checkout root from this script. This avoids stale absolute
# A2S_PROJECT_DIR values after the project is copied to another cluster path.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs/yourmt3_finetune_logs data/reports
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/yourmt3_finetune_logs/task_${STAMP}.log"

{
  echo "A2S project: $PROJECT_DIR"
  echo "Config: $CONFIG"
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
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"

  if [[ -z "${YOURMT3_CUDA_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    case "$CONFIG" in
      *8gpu*) export YOURMT3_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ;;
      *4gpu*) export YOURMT3_CUDA_VISIBLE_DEVICES="0,1,2,3" ;;
    esac
  fi

  echo "Python: $(command -v python)"
  python --version
  python hpc/stage1/check_yourmt3_finetune_data.py --config "$CONFIG"
  python hpc/stage1/launch_yourmt3_finetune.py --config "$CONFIG" --dry_run
  bash hpc/stage1/run_yourmt3_finetune.sh "$CONFIG"

  echo "Finished: $(date -Is)"
} 2>&1 | tee "$LOG"
