#!/usr/bin/env bash
set -euo pipefail

ROOT="${A2S_STAGE1_ROOT:-$PWD/hpc_assets/stage1}"
SOURCE_DIR="${YOURMT3_SOURCE_DIR:-$PWD/third_party/YourMT3}"
ENV_PREFIX="${YOURMT3_ENV_PREFIX:-$ROOT/envs/yourmt3}"
PYTHON_VERSION="${YOURMT3_PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${YOURMT3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
OFFLINE="${YOURMT3_OFFLINE:-0}"
WHEELHOUSE="${YOURMT3_WHEELHOUSE:-}"
MIR_EVAL_SPEC="${YOURMT3_MIR_EVAL_SPEC:-git+https://github.com/craffel/mir_eval.git}"
COSINE_SPEC="${YOURMT3_COSINE_SPEC:-git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup.git}"
USE_CURRENT_ENV="${YOURMT3_USE_CURRENT_ENV:-0}"
REUSE_TORCH="${YOURMT3_REUSE_TORCH:-0}"

if [[ "$USE_CURRENT_ENV" == "1" ]]; then
  test -n "${CONDA_PREFIX:-}" || { echo "Activate the target Conda env first" >&2; exit 2; }
  ENV_PREFIX="$CONDA_PREFIX"
fi

test -f "$SOURCE_DIR/model_helper.py" || { echo "YourMT3 snapshot missing: $SOURCE_DIR" >&2; exit 2; }
command -v conda >/dev/null 2>&1 || { echo "conda is required" >&2; exit 2; }
if [[ ! -x "$ENV_PREFIX/bin/python" && "$OFFLINE" == "1" ]]; then
  echo "Offline mode requires an existing Conda env or an unpacked conda-pack env: $ENV_PREFIX" >&2
  exit 2
elif [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" pip
fi
if [[ "$OFFLINE" != "1" ]]; then
  conda run -p "$ENV_PREFIX" python -m pip install --upgrade pip wheel 'setuptools>=68,<81'
fi

# Install CUDA-matched torch first; override YOURMT3_TORCH_INDEX_URL for the
# center's driver/toolkit. Filter the Space UI's old cu113 torch source and
# browser-only packages so they cannot silently replace the selected CUDA wheel.
FILTERED_ROOT="$ROOT/yourmt3_requirements.filtered.txt"
grep -Ev '^(--extra-index-url|torch([<=> ].*)?$|torchaudio([<=> ].*)?$|yt-dlp|https://github.com/coletdjnz|gradio|gradio_log|spaces|git\+)' \
  "$SOURCE_DIR/requirements.txt" > "$FILTERED_ROOT"
if [[ -f "$SOURCE_DIR/amt/src/requirements.txt" ]]; then
  FILTERED_AMT="$ROOT/yourmt3_amt_requirements.filtered.txt"
  grep -Ev '^(--extra-index-url|torch([<=> ].*)?$|torchaudio([<=> ].*)?$|git\+)' \
    "$SOURCE_DIR/amt/src/requirements.txt" > "$FILTERED_AMT"
fi

if [[ "$OFFLINE" == "1" ]]; then
  test -d "$WHEELHOUSE" || { echo "Set YOURMT3_WHEELHOUSE for offline installation" >&2; exit 2; }
  PIP_OFFLINE=(--no-index --find-links "$WHEELHOUSE")
  if [[ "$REUSE_TORCH" != "1" ]]; then
    conda run -p "$ENV_PREFIX" python -m pip install "${PIP_OFFLINE[@]}" torch torchaudio
  fi
  conda run -p "$ENV_PREFIX" python -m pip install "${PIP_OFFLINE[@]}" -r "$FILTERED_ROOT"
  if [[ -n "${FILTERED_AMT:-}" ]]; then
    conda run -p "$ENV_PREFIX" python -m pip install "${PIP_OFFLINE[@]}" -r "$FILTERED_AMT"
  fi
  conda run -p "$ENV_PREFIX" python -m pip install "${PIP_OFFLINE[@]}" \
    mir_eval cosine-annealing-warmup PyYAML soundfile librosa pretty_midi
else
  if [[ "$REUSE_TORCH" != "1" ]]; then
    conda run -p "$ENV_PREFIX" python -m pip install torch torchaudio --index-url "$TORCH_INDEX_URL"
  fi
  conda run -p "$ENV_PREFIX" python -m pip install -r "$FILTERED_ROOT"
  if [[ -n "${FILTERED_AMT:-}" ]]; then
    conda run -p "$ENV_PREFIX" python -m pip install -r "$FILTERED_AMT"
  fi
  conda run -p "$ENV_PREFIX" python -m pip install \
    "$MIR_EVAL_SPEC" "$COSINE_SPEC" PyYAML soundfile librosa pretty_midi
fi
conda run -p "$ENV_PREFIX" python -m pip freeze > "$ROOT/yourmt3_environment.freeze.txt"
echo "Activate with: conda activate $ENV_PREFIX"
echo "Export YOURMT3_SOURCE_DIR=$SOURCE_DIR"
