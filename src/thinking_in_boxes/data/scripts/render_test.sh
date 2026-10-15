#!/usr/bin/env bash
# =============================================================================
# TEST split — rendering
# =============================================================================
# Produces rgb_{v}.png, bbox_{v}.png and view_{v}.json for every scene.
#
# Depth and the mask outputs are opt-in: add --with-masks to produce them in
# the same pass, or run scripts/render_masks_only.sh afterwards.
#
# Both modes are rendered by default. Pass mode names to render only some;
# any other argument is forwarded to launch.py:
#
#     ./scripts/render_test.sh
#     ./scripts/render_test.sh object_movement
#     ./scripts/render_test.sh --with-masks
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

modes=()
passthrough=()
for arg in "$@"; do
    case "$arg" in
        camera_movement|object_movement) modes+=("$arg") ;;
        *) passthrough+=("$arg") ;;
    esac
done
[[ ${#modes[@]} -eq 0 ]] && modes=(camera_movement object_movement)

for mode in "${modes[@]}"; do
    echo
    echo "=== rendering test split: $mode ==="
    python3 launching/launch.py render --split test --mode "$mode" \
        ${passthrough[@]+"${passthrough[@]}"}
done
