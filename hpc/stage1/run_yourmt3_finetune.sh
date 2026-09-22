#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${YOURMT3_ENV_PREFIX:-${CONDA_PREFIX:?activate the YourMT3 environment first}}"
cd "$PROJECT_DIR"
if [[ -n "${YOURMT3_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$YOURMT3_CUDA_VISIBLE_DEVICES"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
python - <<'PY'
try:
    import torch
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"gpu[{i}]={torch.cuda.get_device_name(i)}")
except Exception as exc:
    print(f"cuda probe failed: {type(exc).__name__}: {exc}")
PY
python hpc/stage1/launch_yourmt3_finetune.py --config "$CONFIG"
