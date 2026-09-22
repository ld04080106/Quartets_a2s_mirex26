#!/usr/bin/env bash
set -euo pipefail

# Optional: make a relocatable environment archive on an Internet-connected
# machine, then unpack it on an offline compute cluster.
ENV_PREFIX="${1:?usage: pack_environment.sh ENV_PREFIX OUTPUT_TAR_GZ}"
OUTPUT="${2:?usage: pack_environment.sh ENV_PREFIX OUTPUT_TAR_GZ}"
command -v conda-pack >/dev/null 2>&1 || {
  echo "conda-pack is required: conda install -c conda-forge conda-pack" >&2
  exit 2
}
conda-pack -p "$ENV_PREFIX" -o "$OUTPUT"
echo "On the cluster: mkdir ENV_DIR && tar -xzf $OUTPUT -C ENV_DIR && ENV_DIR/bin/conda-unpack"
