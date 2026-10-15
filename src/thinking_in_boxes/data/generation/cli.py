# Command line for phase 1 (``blenderproc run generate_scenes.py``).

import os
import sys
import time

import blenderproc as bproc

from config.argv import flag, parse_argv, value
from config.data_root import resolve_data_root
from config.paths import abspath, fail
from config.profiles import VALID_MODES, get_profile
from config.quality import CAMERA_LENS_MM, RENDER_RESOLUTION
from generation.assets import build_asset_index
from generation.scene import generate_one_scene

OPTIONS = {
    "--data-root": value("data_root"),
    "--output-dir": value("output_dir"),
    "--split": value("split"),
    "--start": value("start", int),
    "--count": value("count", int),
    "--num-views": value("num_views", int),
    "--mode": value("mode"),
    "--min-obj": value("min_obj", int),
    "--max-obj": value("max_obj", int),
    # Ablation toggles (used by the ablation scripts)
    "--enable-translation": flag("enable_translation", True),
    "--no-translation": flag("enable_translation", False),
    "--enable-scaling": flag("enable_scaling", True),
    "--no-scaling": flag("enable_scaling", False),
    "--enable-rotation": flag("enable_rotation", True),
    "--no-rotation": flag("enable_rotation", False),
}

# Keys forwarded to the split profile as overrides rather than kept as args.
PROFILE_OVERRIDES = (
    "min_obj",
    "max_obj",
    "enable_translation",
    "enable_scaling",
    "enable_rotation",
)


def parse_args():
    args = parse_argv(sys.argv, OPTIONS)
    overrides = {k: args[k] for k in PROFILE_OVERRIDES if k in args}

    try:
        data_root = resolve_data_root(args.get("data_root"))
    except ValueError as e:
        fail(e)

    split = args.get("split", "train")
    if split not in ("train", "test"):
        fail(f"--split must be 'train' or 'test', got '{split}'")

    mode = args.get("mode")
    if mode is not None:
        if mode not in VALID_MODES:
            fail(f"--mode must be one of {VALID_MODES}, got '{mode}'")
        overrides["mode"] = mode

    output_dir = (
        args.get("output_dir")
        or os.environ.get("OUTPUT_DIR")
        or os.path.join(os.getcwd(), "scene_metadata")
    )
    output_dir = abspath(output_dir)

    profile = get_profile(split, overrides)
    num_views = args.get("num_views") or profile["num_views"]

    return (
        data_root,
        output_dir,
        args.get("start", 0),
        args.get("count", 10),
        num_views,
        profile,
    )


def main():
    t_start = time.time()
    data_root, output_dir, start, count, num_views, profile = parse_args()
    print(f"\n[GENERATE] Scenes {start}..{start + count - 1} ({count} total)")
    print(f"  DATA_ROOT: {data_root}\n  OUTPUT: {output_dir}")
    print(f"  SPLIT: {profile['name']}  MODE: {profile['mode']}")
    print(
        f"  Objects: {profile['min_obj']}..{profile['max_obj']}  "
        f"Views: {num_views}  Target size: {profile['normalize_target_size']}"
    )
    print(
        f"  Perturbation: translation={profile['enable_translation']} "
        f"scaling={profile['enable_scaling']} rotation={profile['enable_rotation']}"
    )

    bproc.init()
    bproc.camera.set_resolution(RENDER_RESOLUTION, RENDER_RESOLUTION)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=CAMERA_LENS_MM, lens_unit="MILLIMETERS"
    )
    index = build_asset_index(data_root)
    if not index["models"]:
        print("[ERROR] No models found")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    done = 0
    for si in range(start, start + count):
        t0 = time.time()
        ok = generate_one_scene(si, data_root, index, num_views, output_dir, profile)
        el = time.time() - t0
        if ok:
            done += 1
            if done <= 3 or done % 50 == 0:
                avg = (time.time() - t_start) / done
                eta = avg * (count - done)
                print(
                    f"  [{done}/{count}] scene_{si:06d} {el:.2f}s "
                    f"(avg {avg:.2f}s, ETA {eta / 60:.0f}min)"
                )
        else:
            print(f"  [{done}/{count}] scene_{si:06d} SKIPPED")

    print(f"\n[DONE] {done}/{count} scenes in {time.time() - t_start:.0f}s")
