# Repository layout: where the entry-point scripts live.

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENERATE_SCRIPT = os.path.join(PROJECT_ROOT, "generation", "generate_scenes.py")
RENDER_SCRIPT = os.path.join(PROJECT_ROOT, "rendering", "render_scenes.py")
DISTRIBUTE_SCRIPT = os.path.join(PROJECT_ROOT, "pipelines", "distribute_generate.py")
LAUNCH_SCRIPT = os.path.join(PROJECT_ROOT, "launching", "launch.py")
