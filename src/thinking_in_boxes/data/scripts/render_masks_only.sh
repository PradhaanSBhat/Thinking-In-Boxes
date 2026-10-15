#!/usr/bin/env bash
# =============================================================================
# Masks-only pass
# =============================================================================
# Regenerates ONLY the per-object mask PNGs (obj_mask_{v}_{o}.png) and the
# tight 2D bounding boxes (bbox_coords_{v}.json), assuming the RGB and bbox
# images already exist.
#
# The RGB/depth passes are skipped entirely: the full-scene depth needed for
# the occlusion test is read back from the existing depth_{v}.png.
#
# If rgb_*.png / depth_*.png are missing, launch.py asks whether to render
# everything together (rgb + depth + bbox + masks); add --yes to answer that
# automatically in non-interactive runs.
#
# Defaults to the test split's object_movement scenes from config.yaml:
#
#     ./scripts/render_masks_only.sh
#     ./scripts/render_masks_only.sh --split train
#     ./scripts/render_masks_only.sh --scene-dir /path/to/scenes --yes
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# --split/--mode may be overridden by the caller; these are just defaults.
args=(--masks-only)
[[ "$*" == *--split* ]] || args+=(--split test)
[[ "$*" == *--mode*  || "$*" == *--scene-dir* ]] || args+=(--mode object_movement)

exec python3 launching/launch.py render "${args[@]}" "$@"
