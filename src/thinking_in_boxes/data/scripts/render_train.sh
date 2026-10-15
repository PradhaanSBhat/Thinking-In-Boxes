#!/usr/bin/env bash
# =============================================================================
# TRAIN split — rendering
# =============================================================================
# Produces rgb_{v}.png, bbox_{v}.png and view_{v}.json for every scene.
#
# Depth and the mask outputs (mask_*, obj_mask_*, bbox_coords_*) are opt-in:
# add --with-masks to produce them in the same pass, or run
# scripts/render_masks_only.sh afterwards.
#
#     ./scripts/render_train.sh
#     ./scripts/render_train.sh --fast --scenes 20      # quick smoke test
#     ./scripts/render_train.sh --with-masks
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

exec python3 launching/launch.py render --split train "$@"
