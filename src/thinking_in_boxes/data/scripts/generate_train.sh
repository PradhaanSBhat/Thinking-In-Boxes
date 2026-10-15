#!/usr/bin/env bash
# =============================================================================
# TRAIN split — scene metadata generation
# =============================================================================
# 4 objects per scene, 2 views, camera moves between views AND a random
# object is perturbed (translation + scaling + rotation).
#
# All settings come from config.yaml. Edit that file, then run:
#
#     ./scripts/generate_train.sh
#
# Any argument you add is passed straight through and overrides the config:
#
#     ./scripts/generate_train.sh --global-count 100          # quick test
#     ./scripts/generate_train.sh --config other_config.yaml
#
# Scene indices are GLOBAL: to shard across machines, give each one a disjoint
# range (set scale.train_start / scale.train_scenes in config.yaml, or pass
# --global-start / --global-count here).
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

exec python3 pipelines/distribute_generate.py --split train "$@"
