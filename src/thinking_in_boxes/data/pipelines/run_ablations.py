#!/usr/bin/env python3
# Single-factor ablation runner.
#
# Builds the ablation datasets declared under `ablations:` in config.yaml.
#
# Usage:
#   python3 run_ablations.py                       # generate + render all
#   python3 run_ablations.py only_rotate           # just one
#   python3 run_ablations.py --skip-render         # metadata only
#   python3 run_ablations.py --dry-run             # print commands only

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.ablations import main  # noqa: E402

if __name__ == "__main__":
    main()
