#!/usr/bin/env bash
# =============================================================================
# Phase 0 — download the Objaverse-XL models listed by ARSG-110K
# =============================================================================
# No longer needs a clone of 3D-Fixer: the part of its dataset toolkit this
# pipeline uses is vendored under preparation/objaverse/ (see
# preparation/THIRD_PARTY_NOTICES.md).
#
# Models land directly in the configured asset root, at
#   <data_root>/<assets.dataset>/raw/hf-objaverse-v1/glbs/
# which is where ./scripts/convert_assets.sh reads from — nothing to move.
#
#     ./scripts/download_objaverse.sh                       # the whole listing
#     ./scripts/download_objaverse.sh --world-size 16       # one sixteenth
#     ./scripts/download_objaverse.sh --world-size 16 --rank 3
#     ./scripts/download_objaverse.sh --metadata-only       # just the listing
#
# Needs: pip install objaverse pandas huggingface_hub fsspec
#
# Long transfers drop, and the downloader can die outright (killed worker,
# segfault, OOM) rather than raising, so the whole process is re-run until it
# exits cleanly. Each attempt skips whatever is already on disk. Usage errors
# (exit 2) and Ctrl-C (exit 130) stop the loop instead of retrying.
#
#     RETRY_DELAY=30 ./scripts/download_objaverse.sh     # seconds between runs
#
# Then convert the .glb files to .blend:
#
#     ./scripts/convert_assets.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

RETRY_DELAY="${RETRY_DELAY:-10}"

echo "Starting resilient download loop..."

attempt=1
while true; do
    echo "Running download (attempt $attempt)..."
    exit_code=0
    python3 preparation/download_objaverse.py "$@" || exit_code=$?

    case "$exit_code" in
        0)
            echo "Download finished successfully!"
            break
            ;;
        2|130)
            echo "Download stopped (exit code $exit_code), not retrying."
            exit "$exit_code"
            ;;
        *)
            echo "Download crashed with exit code $exit_code"
            echo "Retrying in $RETRY_DELAY seconds..."
            sleep "$RETRY_DELAY"
            attempt=$((attempt + 1))
            ;;
    esac
done
