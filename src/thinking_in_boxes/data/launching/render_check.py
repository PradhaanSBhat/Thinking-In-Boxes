# Prerequisite checks for the --masks-only render pass.

import os
import sys


def check_render_prerequisites(scene_dir, start, scenes):
    """Count how many scenes already have RGB + bbox renders.

    Returns (n_scenes_with_metadata, n_scenes_with_rgb, n_scenes_with_bbox).
    """
    n_meta = n_rgb = n_bbox = 0
    for idx in range(start, start + scenes):
        sd = os.path.join(scene_dir, f"scene_{idx:06d}")
        if not os.path.isfile(os.path.join(sd, "metadata.json")):
            continue
        n_meta += 1
        if os.path.isfile(os.path.join(sd, "rgb_0.png")):
            n_rgb += 1
        if os.path.isfile(os.path.join(sd, "bbox_0.png")):
            n_bbox += 1
    return n_meta, n_rgb, n_bbox


def confirm_full_render(n_meta, n_rgb, n_bbox, assume_yes=False):
    """Ask the user whether to fall back to a full render.

    Returns True to run a full render (rgb + bbox + masks), False to abort.
    """
    print(
        "\n[WARN] --masks-only was requested, but the RGB / bbox renders "
        "it fills in from are missing."
    )
    print(f"  scenes with metadata  : {n_meta}")
    print(f"  scenes with rgb_0.png : {n_rgb}")
    print(f"  scenes with bbox_0.png: {n_bbox}")
    print(
        "\nThe masks-only pass adds mask outputs to scenes that have already "
        "been rendered;\nthere is nothing to add to a scene with no renders."
    )

    if assume_yes:
        print("\n[--yes] Proceeding with a FULL render (rgb + bbox + depth + masks).")
        return True

    if not sys.stdin.isatty():
        print(
            "\n[ERROR] Not an interactive terminal and --yes was not given. "
            "Aborting.\n        Re-run with --yes to generate everything, "
            "or drop --masks-only."
        )
        return False

    print(
        "\nGenerate everything together instead (rgb + bbox + depth + masks)? [y/N] ",
        end="",
        flush=True,
    )
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")
