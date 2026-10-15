#!/usr/bin/env python3
# Unified launcher for generate + render.
#
# Continuous-pool scheduler: keeps exactly `max_concurrent` workers running at
# all times.  When any single worker finishes, the next task starts immediately,
# so there is no GPU idle time at wave boundaries.
#
# Generate:
#   python3 launch.py generate --split train \
#       --data-root /path/to/datasets --output-dir ./scenes_train \
#       --scenes 20000 --workers 40
#
# Render (default — rgb + bbox + view JSON only):
#   python3 launch.py render --split train \
#       --data-root /path/to/datasets --scene-dir ./scenes_train
#
# Render everything in one pass (adds depth + all masks):
#   python3 launch.py render --split train --with-masks ...
#
# Render masks only (rgb + bbox must already exist):
#   python3 launch.py render --split test --masks-only \
#       --data-root /path/to/datasets --scene-dir ./scenes_test
#
# By default the render command produces only rgb_*.png, bbox_*.png and
# view_*.json.  Depth and every mask output are written by --with-masks, or by
# a separate --masks-only pass over already-rendered scenes.  If --masks-only
# finds no RGB/bbox renders to fill in, the launcher asks whether to run a full
# render instead.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from launching.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
