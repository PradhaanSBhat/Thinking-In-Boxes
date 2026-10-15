# Render from Metadata  (train + test).
#
# Phase 2: renders the scenes described by scene_NNNNNN/metadata.json.
#
# Default outputs per view:
#   rgb_{v}.png            — HDRI-lit RGB render
#   bbox_{v}.png           — 3D bounding-box overlay on checkerboard floor
#   view_{v}.json          — camera intrinsics/extrinsics + object metadata
#
# Mask outputs, written only when masks are enabled:
#   depth_{v}.png          — uint16 depth in millimetres
#   mask_{v}_{o}.png       — per-object bbox-face mask (from the bbox pass)
#   obj_mask_{v}_{o}.png   — pixel-accurate per-object mask
#   bbox_coords_{v}.json   — tight 2D bounding boxes from full silhouettes
#
# Render passes:
#   (default)     RGB + bbox + view JSON only.  No depth, no masks.
#   --with-masks  Full render plus every mask output listed above.
#   --masks-only  Skip the RGB and bbox passes and generate ONLY the mask
#                 outputs, for scenes whose rgb_*.png / bbox_*.png already
#                 exist.  The full-scene depth needed for occlusion testing is
#                 rendered here (1 sample, depth pass only) rather than being
#                 read back from disk.
#
# The asset root is taken from --data-root, or from the DATA_ROOT environment
# variable if that flag is omitted.  There is no built-in default path.
#
# Usage:
#   blenderproc run render_scenes.py \
#       --split train \
#       --data-root /path/to/datasets \
#       --scene-dir ./scenes --start 0 --count 1000
#
#   # or, with DATA_ROOT exported:
#   export DATA_ROOT=/path/to/datasets
#   blenderproc run render_scenes.py \
#       --split test --masks-only \
#       --scene-dir ./scenes_test --start 0 --count 500

import blenderproc as bproc  # noqa: F401  (must be the first import)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
