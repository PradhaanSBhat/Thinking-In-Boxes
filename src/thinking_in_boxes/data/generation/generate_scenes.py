# Scene Metadata Generator  (train + test).
#
# Phase 1: generates scene_NNNNNN/metadata.json files with RELATIVE paths.
# All placement, collision checks and camera visibility are computed here —
# no rendering happens in this phase.
#
# The asset root is taken from --data-root, or from the DATA_ROOT environment
# variable if that flag is omitted.  There is no built-in default path.
#
# Usage:
#   blenderproc run generate_scenes.py \
#       --split train \
#       --data-root /path/to/datasets \
#       --output-dir ./scenes --start 0 --count 1000
#
#   # or, with DATA_ROOT exported:
#   export DATA_ROOT=/path/to/datasets
#   blenderproc run generate_scenes.py \
#       --split test --mode object_movement \
#       --output-dir ./scenes_test --start 0 --count 500
#
# Modes (test split):
#   camera_movement  — camera moves between views, objects stay fixed
#   object_movement  — objects move between views, camera stays fixed
#
# The train split uses mode "mixed": the camera moves between views AND a
# random object is perturbed per view.  See dsgen/config/profiles.py for the
# full set of per-split differences.

import blenderproc as bproc  # noqa: F401  (must be the first import)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
