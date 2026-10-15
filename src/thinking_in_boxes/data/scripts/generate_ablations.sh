#!/usr/bin/env bash
# =============================================================================
# Single-factor ablation test sets
# =============================================================================
# Builds the datasets declared under `ablations:` in config.yaml. Each isolates
# one kind of object variation (translation / scaling / rotation) so a model's
# sensitivity to it can be measured independently.
#
# Edit the `ablations:` section of config.yaml to add or change them.
#
#     ./scripts/generate_ablations.sh                  # all of them
#     ./scripts/generate_ablations.sh only_rotate      # just one
#     ./scripts/generate_ablations.sh --skip-render    # metadata only
#     ./scripts/generate_ablations.sh --dry-run
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

exec python3 pipelines/run_ablations.py "$@"
