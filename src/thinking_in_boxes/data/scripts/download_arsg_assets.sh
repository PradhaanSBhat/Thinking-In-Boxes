#!/usr/bin/env bash
# =============================================================================
# Download the ARSG-110K assets: HDRIs + floor/wall materials
# =============================================================================
# Fetches from https://huggingface.co/datasets/HorizonRobotics/ARSG-110K
# and unzips into the layout the scene generator expects:
#
#     <data_root>/ARSG-110K/hdrs/             <- required
#     <data_root>/ARSG-110K/materials_floor/  <- optional
#     <data_root>/ARSG-110K/materials_wall/   <- optional
#
# HDRIs are REQUIRED: every scene picks one for lighting, and generation fails
# without them. Materials are optional — scenes fall back to a plain grey
# floor if none are found.
#
# These are large: hdrs 7.5 GB, materials_floor 8.3 GB, materials_wall 16 GB.
#
#     ./scripts/download_arsg_assets.sh              # all three
#     ./scripts/download_arsg_assets.sh hdrs         # just the HDRIs
#     ./scripts/download_arsg_assets.sh hdrs materials_floor
#     KEEP_ZIPS=1 ./scripts/download_arsg_assets.sh  # don't delete archives
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

REPO="https://huggingface.co/datasets/HorizonRobotics/ARSG-110K/resolve/main"
KEEP_ZIPS="${KEEP_ZIPS:-0}"

# Resolve data_root from config.yaml (DATA_ROOT still overrides it).
# Assigned separately from the declaration so a non-zero exit is not masked.
DATA_ROOT=""
if ! DATA_ROOT="$(python3 -c "
import sys
sys.path.insert(0, '.')
from config import load_settings
try:
    print(load_settings().require_data_root())
except ValueError as e:
    sys.stderr.write(f'[ERROR] {e}\n'); sys.exit(1)
")"; then
    exit 1
fi

DEST="$DATA_ROOT/ARSG-110K"
mkdir -p "$DEST"

components=("$@")
[[ ${#components[@]} -eq 0 ]] && components=(hdrs materials_floor materials_wall)

echo "─────────────────────────────────────────────"
echo "  Destination: $DEST"
echo "  Components : ${components[*]}"
echo "─────────────────────────────────────────────"

for name in "${components[@]}"; do
    case "$name" in
        hdrs|materials_floor|materials_wall) ;;
        *) echo "[ERROR] unknown component '$name' (expected hdrs, "\
                "materials_floor or materials_wall)" >&2; exit 1 ;;
    esac

    if [[ -d "$DEST/$name" ]] && [[ -n "$(ls -A "$DEST/$name" 2>/dev/null)" ]]; then
        echo "[SKIP] $name already present at $DEST/$name"
        continue
    fi

    zip_path="$DEST/$name.zip"

    if [[ ! -f "$zip_path" ]]; then
        echo
        echo "=== downloading $name.zip ==="
        # -C - resumes a partial download if the connection dropped.
        curl -L -C - --fail --retry 5 --retry-delay 10 \
             -o "$zip_path" "$REPO/$name.zip"
    else
        echo "[SKIP] $name.zip already downloaded"
    fi

    echo "=== unzipping $name.zip ==="
    unzip -q -o "$zip_path" -d "$DEST"

    [[ "$KEEP_ZIPS" == "1" ]] || rm -f "$zip_path"
done

echo
echo "─────────────────────────────────────────────"
echo "  Done. Contents of $DEST:"
ls -1 "$DEST" | sed 's/^/    /'
echo "─────────────────────────────────────────────"
