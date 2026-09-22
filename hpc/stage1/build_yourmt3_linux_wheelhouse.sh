#!/usr/bin/env bash
set -euo pipefail

# Run only on an Internet-connected Linux host compatible with the cluster.
# A Windows wheelhouse cannot be used on a Linux supercomputer.
SOURCE_DIR="${YOURMT3_SOURCE_DIR:?set YOURMT3_SOURCE_DIR to the unpacked Space snapshot}"
WHEELHOUSE="${YOURMT3_WHEELHOUSE:-$PWD/yourmt3-wheelhouse}"
TORCH_INDEX_URL="${YOURMT3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$WHEELHOUSE"

FILTERED_ROOT="$WHEELHOUSE/requirements.filtered.txt"
grep -Ev '^(--extra-index-url|torch([<=> ].*)?$|torchaudio([<=> ].*)?$|yt-dlp|https://github.com/coletdjnz|gradio|gradio_log|spaces|git\+)' \
  "$SOURCE_DIR/requirements.txt" > "$FILTERED_ROOT"
"$PYTHON_BIN" -m pip wheel --wheel-dir "$WHEELHOUSE" \
  torch torchaudio --index-url "$TORCH_INDEX_URL"
"$PYTHON_BIN" -m pip wheel --wheel-dir "$WHEELHOUSE" -r "$FILTERED_ROOT"
"$PYTHON_BIN" -m pip wheel --wheel-dir "$WHEELHOUSE" \
  git+https://github.com/craffel/mir_eval.git \
  git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup.git \
  PyYAML soundfile librosa pretty_midi

if [[ -f "$SOURCE_DIR/amt/src/requirements.txt" ]]; then
  FILTERED_AMT="$WHEELHOUSE/amt_requirements.filtered.txt"
  grep -Ev '^(--extra-index-url|torch([<=> ].*)?$|torchaudio([<=> ].*)?$|git\+)' \
    "$SOURCE_DIR/amt/src/requirements.txt" > "$FILTERED_AMT"
  "$PYTHON_BIN" -m pip wheel --wheel-dir "$WHEELHOUSE" -r "$FILTERED_AMT"
fi
find "$WHEELHOUSE" -type f -name '*.whl' -print0 | sort -z | xargs -0 sha256sum > "$WHEELHOUSE/SHA256SUMS"
echo "Linux wheelhouse ready: $WHEELHOUSE"
