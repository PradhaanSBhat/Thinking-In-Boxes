#!/usr/bin/env python3
# Single-machine generation distributor.
#
# Splits a GLOBAL scene range across --num-workers local blenderproc processes.
#
# Usage:
#   python3 distribute_generate.py --split train \
#       --data-root /path/to/datasets \
#       --output-dir scenes_train \
#       --global-start 0 --global-count 20000 --num-workers 40
#
#   python3 distribute_generate.py --split test --mode object_movement \
#       --data-root /path/to/datasets \
#       --output-dir scenes_test \
#       --global-start 0 --global-count 500 --num-workers 40
#
# For multi-GPU rendering use launch.py instead.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.distribute import main  # noqa: E402

if __name__ == "__main__":
    main()
