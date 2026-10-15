# GLB → .blend asset converter (phase 0, run once).
#
# Usage:
#   blenderproc run convert_glb_to_blend.py \
#       --input-root  /path/to/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs \
#       --output-root /path/to/ObjaverseXL_sketchfab/converted
#
#   # Non-recursive layouts (flat directory of .glb files):
#   blenderproc run convert_glb_to_blend.py \
#       --input-root ./glbs --output-root ./converted --glob "*.glb"
#
#   # --prune deletes each source .glb once its .blend is written, so only one
#   # copy of each model is left on disk.

import blenderproc as bproc  # noqa: F401  (must be the first import)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.convert import main  # noqa: E402

if __name__ == "__main__":
    main()
