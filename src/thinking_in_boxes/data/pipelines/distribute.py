# Single-machine generation distributor.
#
# Splits a GLOBAL scene range across NUM_WORKERS local blenderproc processes.
#
# The global range is what makes multi-machine sharding work: each machine is
# given a disjoint --global-start / --global-count, and every worker on that
# machine computes its own absolute scene indices from it.  Scene numbering is
# therefore consistent across the whole cluster.
#
# For multi-GPU rendering use launch.py instead.

import argparse
import math
import os
import subprocess
import sys
import time

from config.paths import abspath
from config.profiles import get_profile, normalize_generator_flags
from config.project import GENERATE_SCRIPT
from config.settings import CONFIG_FILENAME, load_settings


def build_parser():
    parser = argparse.ArgumentParser(
        description=f"Fan phase-1 generation across local workers. Defaults come "
        f"from {CONFIG_FILENAME}; any argument here overrides it."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=f"Configuration file (default: {CONFIG_FILENAME} next to this script)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Asset dataset root (default: paths.data_root)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where scene metadata is written "
        "(default: the split's configured scene directory)",
    )

    # SPLIT SELECTION
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which dataset split profile to use (default: train)",
    )

    # MODE: mixed (train default), camera_movement or object_movement
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["mixed", "camera_movement", "object_movement"],
        help="Override the split's default movement mode. "
        "mixed: camera moves and a random object is perturbed. "
        "camera_movement: only camera moves between views, objects fixed. "
        "object_movement: only objects move between views, camera fixed.",
    )

    # GLOBAL RANGE (important)
    parser.add_argument(
        "--global-start",
        type=int,
        default=None,
        help="First global scene index (default: scale.train_start)",
    )
    parser.add_argument(
        "--global-count",
        type=int,
        default=None,
        help="How many scenes (default: scale.train_scenes / scale.test_scenes)",
    )

    # LOCAL PARALLELISM
    parser.add_argument(
        "--min-obj",
        type=int,
        default=None,
        help="Min objects per scene (default: objects.<split>_min)",
    )
    parser.add_argument(
        "--max-obj",
        type=int,
        default=None,
        help="Max objects per scene (default: objects.<split>_max)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers (default: scale.num_workers)",
    )

    # LOGGING
    parser.add_argument(
        "--log-dir", type=str, default=None, help="Default: <output-dir>/logs"
    )

    # EXTRA GENERATOR FLAGS (ablations)
    parser.add_argument(
        "--generator-flag",
        action="append",
        default=None,
        metavar="FLAG",
        help="Extra flag passed through to generate_scenes.py "
        "(repeatable). Use the '=' form so argparse does not "
        "mistake the value for an option of its own, e.g. "
        "--generator-flag=--no-scaling",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Resolve settings: CLI argument > config.yaml > built-in default ---
    try:
        cfg = load_settings(args.config)
        data_root = (
            abspath(args.data_root) if args.data_root else cfg.require_data_root()
        )
    except ValueError as e:
        parser.error(str(e))

    split = args.split

    output_dir = args.output_dir or cfg.scene_dir_for(split, args.mode)
    output_dir = abspath(output_dir)

    global_start = (
        args.global_start if args.global_start is not None else cfg.start_for(split)
    )
    global_count = (
        args.global_count if args.global_count is not None else cfg.scenes_for(split)
    )
    num_workers = (
        args.num_workers if args.num_workers is not None else cfg.scale["num_workers"]
    )

    # Validate the split/mode combination and pass-through flags before
    # launching anything.
    try:
        profile = get_profile(split, {"mode": args.mode} if args.mode else None)
        generator_flags = normalize_generator_flags(args.generator_flag)
        min_obj, max_obj = cfg.object_range(split, args.min_obj, args.max_obj)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    mode = profile["mode"]

    log_dir = args.log_dir or os.path.join(cfg.paths["log_root"], "generate")
    os.makedirs(log_dir, exist_ok=True)

    scenes_per_worker = math.ceil(global_count / num_workers)

    print(f"[Distribute] Split: {split}")
    print(f"[Distribute] Mode: {mode}")
    print(f"[Distribute] Objects per scene: {min_obj}..{max_obj}")
    print(
        f"[Distribute] Global range: {global_start} → {global_start + global_count - 1}"
    )
    print(f"[Distribute] Workers: {num_workers} ({scenes_per_worker} scenes each)")
    print(f"[Distribute] Logs: {log_dir}")

    procs = []

    for i in range(num_workers):
        local_start = i * scenes_per_worker
        count = min(scenes_per_worker, global_count - local_start)

        if count <= 0:
            break

        # THIS IS THE KEY: local worker offset → absolute scene index
        worker_start = global_start + local_start

        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"worker_{i}_{ts}.log")
        log_file = open(log_path, "w")

        cmd = [
            "blenderproc",
            "run",
            GENERATE_SCRIPT,
            "--split",
            split,
            "--data-root",
            data_root,
            "--output-dir",
            output_dir,
            "--start",
            str(worker_start),
            "--count",
            str(count),
            "--mode",
            mode,
            "--min-obj",
            str(min_obj),
            "--max-obj",
            str(max_obj),
        ]
        cmd.extend(generator_flags)

        print(
            f"[Machine Chunk] Worker {i}: scenes {worker_start} → "
            f"{worker_start + count - 1} (split={split}, mode={mode})"
        )
        print(f"  Log: {log_path}")

        p = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        procs.append((p, log_file))

    # Wait
    failed = 0
    for p, log_file in procs:
        ret = p.wait()
        log_file.close()

        if ret != 0:
            failed += 1
            print("Worker failed, check logs.")

    print("Done")
    if failed:
        sys.exit(1)
