#!/usr/bin/env bash
# =============================================================================
# Asset preprocessing: GLB → .blend
# =============================================================================
# Run this ONCE before generating any scenes.
#
# Paths come from the `assets:` section of config.yaml. The scene generator
# only indexes .blend files under a `converted/` directory, so
# assets.blend_output should end in `converted`.
#
#     ./scripts/convert_assets.sh
#     ./scripts/convert_assets.sh --prune            # delete each .glb once converted
#     ./scripts/convert_assets.sh --overwrite
#     ./scripts/convert_assets.sh --glob "*.glb"     # flat input directory
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

exec blenderproc run pipelines/convert_glb_to_blend.py "$@"
