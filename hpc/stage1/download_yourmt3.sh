#!/usr/bin/env bash
set -euo pipefail

# Downloads the official-author Hugging Face Space snapshot, which contains
# the inference code and checkpoint layout used by the demo.
ROOT="${A2S_STAGE1_ROOT:-$PWD/hpc_assets/stage1}"
SOURCE_DIR="${YOURMT3_SOURCE_DIR:-$PWD/third_party/YourMT3}"
REVISION="${YOURMT3_REVISION:-5e66c1ea173a8186e0d20432b841d3180cc015b5}"
mkdir -p "$SOURCE_DIR"

if command -v hf >/dev/null 2>&1; then
  if ! hf download mimbres/YourMT3 --repo-type space --revision "$REVISION" \
    --exclude "amt/logs/**" "amt/src/extras/**" "amt/src/tests/**" \
      "**/__pycache__/**" "*.pyc" ".coverage" ".DS_Store" \
      "amt/src/utils/preprocess/preprocess_rnsynth.py" \
    --local-dir "$SOURCE_DIR"; then
    echo "ERROR: Hugging Face download failed." >&2
    echo "If this node has no external network, use create_yourmt3_bundle.py locally," >&2
    echo "upload the tar.gz, then run import_yourmt3_bundle.py on the cluster." >&2
    exit 3
  fi
else
  echo "ERROR: Hugging Face hf CLI is required (pip install huggingface_hub)." >&2
  exit 2
fi

printf '%s\n' "$REVISION" > "$ROOT/yourmt3_requested_revision.txt"
find "$SOURCE_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$ROOT/yourmt3_snapshot.sha256"
echo "YourMT3 Space snapshot: $SOURCE_DIR"
