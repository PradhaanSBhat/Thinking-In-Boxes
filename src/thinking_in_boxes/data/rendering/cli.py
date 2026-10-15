# Command line for phase 2 (``blenderproc run render_scenes.py``).

import os
import sys
import time

import blenderproc as bproc
import numpy as np

from blender.renderer import init_renderer, purge_orphan_data
from config.argv import flag, parse_argv, value
from config.data_root import resolve_data_root
from config.paths import abspath, fail
from config.profiles import get_profile
from config.quality import (
    FLUSH_INTERVAL,
    ORPHAN_PURGE_INTERVAL,
    OUTPUT_RESOLUTION,
    QUALITY_FAST,
    QUALITY_NORMAL,
)
from rendering.passes import (
    render_scene,
    render_scene_masks_only,
    scene_has_masks,
    scene_is_rendered,
)
from rendering.writer import AsyncWriter

OPTIONS = {
    "--data-root": value("data_root"),
    "--scene-dir": value("scene_dir"),
    "--split": value("split"),
    "--start": value("start", int),
    "--count": value("count", int),
    "--fast": flag("fast"),
    "--masks-only": flag("masks_only"),
    "--with-masks": flag("with_masks"),
    "--no-masks": flag("no_masks"),
}

DEFAULTS = {
    "data_root": None,
    "scene_dir": None,
    "split": "train",
    "start": 0,
    "count": 10,
    "fast": False,
    "masks_only": False,
    "with_masks": False,
    "no_masks": False,
}


def parse_args():
    args = {**DEFAULTS, **parse_argv(sys.argv, OPTIONS)}

    try:
        data_root = resolve_data_root(args["data_root"])
    except ValueError as e:
        fail(e)

    scene_dir = args["scene_dir"] or os.environ.get("SCENE_DIR")
    if not scene_dir:
        fail("--scene-dir required (or set SCENE_DIR)")
    scene_dir = abspath(scene_dir)

    split = args["split"]
    if split not in ("train", "test"):
        fail(f"--split must be 'train' or 'test', got '{split}'")
    if args["masks_only"] and args["no_masks"]:
        fail("--masks-only and --no-masks are mutually exclusive")

    # Mask outputs (depth + every mask file) are OFF by default for both
    # splits: a plain render produces rgb / bbox / view JSON only.  They are
    # enabled explicitly with --with-masks, or produced by a --masks-only
    # pass over scenes that already have their RGB and bbox renders.
    emit_masks = (args["with_masks"] and not args["no_masks"]) or args["masks_only"]

    return (
        data_root,
        scene_dir,
        args["start"],
        args["count"],
        args["fast"],
        args["masks_only"],
        emit_masks,
        get_profile(split),
    )


def main():
    t_start = time.time()
    (data_root, scene_dir, start, count, fast, masks_only, emit_masks, profile) = (
        parse_args()
    )

    quality = QUALITY_FAST if fast else QUALITY_NORMAL
    output_res = OUTPUT_RESOLUTION

    qmode = "FAST" if fast else "NORMAL"
    pass_name = "MASKS-ONLY" if masks_only else "FULL"
    print(
        f"\n[RENDER] {pass_name} | {qmode} | "
        f"{quality['resolution']}px @ {quality['samples']}spp"
    )
    print(f"  SPLIT: {profile['name']}")
    print(f"  DATA_ROOT: {data_root}")
    print(f"  SCENE_DIR: {scene_dir}")
    print(f"  Range: {start}..{start + count - 1}")
    if not masks_only:
        print(f"  Per-object masks: {'yes' if emit_masks else 'no'}")

    bproc.init()
    init_renderer(quality)

    writer = AsyncWriter(max_workers=4)
    done = 0
    missing = 0
    scene_times = []

    for scene_idx in range(start, start + count):
        sd = os.path.join(scene_dir, f"scene_{scene_idx:06d}")

        if not os.path.isdir(sd):
            continue
        if not os.path.isfile(os.path.join(sd, "metadata.json")):
            continue

        # Skip work that is already on disk, so an interrupted run resumes.
        if masks_only:
            if scene_has_masks(sd):
                done += 1
                continue
        elif scene_is_rendered(sd):
            done += 1
            continue

        t0 = time.time()

        if done > 0 and done % ORPHAN_PURGE_INTERVAL == 0:
            purge_orphan_data()

        if masks_only:
            ok = render_scene_masks_only(
                sd, data_root, quality, writer, profile, output_res
            )
        else:
            ok = render_scene(
                sd, data_root, quality, writer, profile, emit_masks, output_res
            )

        # Flush periodically instead of every scene
        if done % FLUSH_INTERVAL == 0:
            writer.flush()

        elapsed = time.time() - t0

        if ok == "missing":
            missing += 1
            if missing <= 5:
                print(
                    f"  scene_{scene_idx:06d} SKIPPED "
                    f"(rgb/depth not found — run a full render first)"
                )
            continue

        if ok:
            done += 1
            scene_times.append(elapsed)
            if done <= 3 or done % 10 == 0:
                avg = np.mean(scene_times[-50:])
                eta = avg * (count - done)
                print(
                    f"  [{done}/{count}] scene_{scene_idx:06d} "
                    f"{elapsed:.1f}s (avg {avg:.1f}s, ETA {eta / 60:.0f}min)"
                )
        else:
            print(f"  scene_{scene_idx:06d} FAILED")

    writer.shutdown()
    total = time.time() - t_start
    print(f"\n[DONE] {done} scenes in {total:.0f}s ({total / max(done, 1):.1f}s/scene)")
    if missing:
        print(
            f"[WARN] {missing} scenes skipped: RGB/depth renders missing. "
            f"Run without --masks-only to generate them."
        )
