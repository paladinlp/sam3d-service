#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_TAG="${SAM3D_CHECKPOINT_TAG:-hf}"
PIPELINE_CONFIG="$ROOT_DIR/checkpoints/$CHECKPOINT_TAG/pipeline.yaml"

if [[ ! -f "$PIPELINE_CONFIG" ]]; then
  echo "Missing pipeline config: $PIPELINE_CONFIG" >&2
  echo "Download the official checkpoints before starting the service." >&2
  exit 1
fi

python -m sam3d_service.main
