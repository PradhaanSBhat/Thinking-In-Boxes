# The ``generate`` and ``render`` subcommands of launch.py.

import os
import sys
import time

from config.profiles import get_profile, normalize_generator_flags
from config.project import GENERATE_SCRIPT, RENDER_SCRIPT
from launching.gpu import detect_gpus, gpu_info
from launching.pool import _setup_signal_handler, report, run_pool
from launching.render_check import check_render_prerequisites, confirm_full_render


def _resolve_script(override, default):
    script = os.path.abspath(override or default)
    if not os.path.isfile(script):
        print(f"[ERROR] {script} not found")
        sys.exit(1)
    return script


def _batches(start, total, batch_size):
    """(worker_id, offset, count) for each blenderproc launch."""
    for wid, offset in enumerate(range(start, start + total, batch_size)):
        yield wid, offset, min(batch_size, start + total - offset)


def _announce(header, tasks, total, batch_size, max_concurrent):
    print(
        f"\n{header} | {total} scenes, {len(tasks)} batches "
        f"(batch_size={batch_size}), max {max_concurrent} concurrent"
    )
    for task in tasks:
        print(f"  {task['label']}")


def _run(args, tasks, max_concurrent, check, t0=None):
    _setup_signal_handler()
    t0 = time.time() if t0 is None else t0
    results = run_pool(
        tasks,
        max_concurrent,
        args.log_dir,
        args.stagger,
        check,
        refresh=args.monitor_sec,
    )
    report(results, t0)


def cmd_generate(args):
    script = _resolve_script(args.script, GENERATE_SCRIPT)

    # Validate the split/mode combination and any pass-through flags up front.
    try:
        profile = get_profile(args.split, {"mode": args.mode} if args.mode else None)
        generator_flags = normalize_generator_flags(args.generator_flag)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    tasks = []
    for wid, offset, count in _batches(args.start, args.scenes, args.batch_size):
        cmd = [
            "blenderproc", "run", script,
            "--split", args.split,
            "--data-root", args.data_root,
            "--output-dir", args.output_dir,
            "--start", str(offset),
            "--count", str(count),
        ]  # fmt: skip
        if args.num_views:
            cmd += ["--num-views", str(args.num_views)]
        cmd += ["--min-obj", str(args.min_obj), "--max-obj", str(args.max_obj)]
        if args.mode:
            cmd += ["--mode", args.mode]
        cmd += generator_flags

        tasks.append(
            {
                "cmd": cmd,
                "label": f"gen_w{wid:02d} [{offset}..{offset + count - 1}]",
                "log_name": f"gen_w{wid:02d}.log",
            }
        )

    _announce(
        f"[GENERATE] split={profile['name']} mode={profile['mode']} | "
        f"objects {args.min_obj}..{args.max_obj}",
        tasks,
        args.scenes,
        args.batch_size,
        args.workers,
    )
    if args.dry_run:
        print("DRY RUN")
        return

    def check():
        try:
            scenes = [
                d
                for d in os.listdir(args.output_dir)
                if d.startswith("scene_")
                and os.path.isfile(os.path.join(args.output_dir, d, "metadata.json"))
            ]
            return f" | {len(scenes)} scenes generated"
        except Exception:
            return ""

    _run(args, tasks, args.workers, check)


def _resolve_masks_only(args):
    """Check the --masks-only prerequisites, offering a full render instead.

    Returns True to stay in masks-only mode; it does not return if the user
    declines, and flips args to a full render with masks if they accept.
    """
    n_meta, n_rgb, n_bbox = check_render_prerequisites(
        args.scene_dir, args.start, args.scenes
    )
    if n_meta == 0:
        print(
            f"[ERROR] No scene metadata found under {args.scene_dir} "
            f"in range {args.start}..{args.start + args.scenes - 1}.\n"
            f"        Run `launch.py generate` first."
        )
        sys.exit(1)

    if n_rgb >= n_meta and n_bbox >= n_meta:
        return True

    if not confirm_full_render(n_meta, n_rgb, n_bbox, args.yes):
        print("[ABORT] Nothing to do.")
        sys.exit(1)

    # Fall back to a full render, keeping the masks the user asked for.
    args.with_masks = True
    args.no_masks = False
    return False


def _resolve_gpus(args):
    if args.gpu_ids:
        gpus = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpus = detect_gpus()[: args.gpus] if args.gpus > 0 else detect_gpus()
    if not gpus:
        print("[ERROR] No GPUs")
        sys.exit(1)

    print(f"GPUs: {gpus}")
    for gid, (used, total) in gpu_info().items():
        if gid in gpus:
            print(f"  GPU {gid}: {used}/{total} MB")
    return gpus


def cmd_render(args):
    script = _resolve_script(args.script, RENDER_SCRIPT)
    profile = get_profile(args.split)

    # The masks-only pass fills in the mask outputs for scenes that already
    # have their RGB and bbox renders.  If those are absent there is nothing
    # to fill in, so ask whether to render everything instead.
    masks_only = args.masks_only and _resolve_masks_only(args)

    gpus = _resolve_gpus(args)
    max_concurrent = len(gpus) * args.procs_per_gpu

    tasks = []
    for wid, offset, count in _batches(args.start, args.scenes, args.batch_size):
        gid = gpus[wid % len(gpus)]
        cmd = [
            "blenderproc", "run", script,
            "--split", args.split,
            "--data-root", args.data_root,
            "--scene-dir", args.scene_dir,
            "--start", str(offset),
            "--count", str(count),
        ]  # fmt: skip
        if args.fast:
            cmd.append("--fast")
        if masks_only:
            cmd.append("--masks-only")
        elif args.with_masks:
            cmd.append("--with-masks")
        elif args.no_masks:
            cmd.append("--no-masks")

        tasks.append(
            {
                "cmd": cmd,
                "label": f"rnd_w{wid:02d}_gpu{gid} [{offset}..{offset + count - 1}]",
                "log_name": f"rnd_w{wid:02d}_gpu{gid}.log",
                "env": {"CUDA_VISIBLE_DEVICES": str(gid)},
            }
        )

    _announce(
        f"[RENDER] {'MASKS-ONLY' if masks_only else 'FULL'} | split={profile['name']}",
        tasks,
        args.scenes,
        args.batch_size,
        max_concurrent,
    )
    if args.dry_run:
        print("DRY RUN")
        return

    # Progress probe: masks-only writes obj_mask_*, a full render writes rgb_1.
    probe = "obj_mask_0_0.png" if masks_only else "rgb_1.png"
    t0 = time.time()

    def check():
        try:
            done = sum(
                os.path.isfile(os.path.join(args.scene_dir, f"scene_{idx:06d}", probe))
                for idx in range(args.start, args.start + args.scenes)
            )
            rate = done / max(time.time() - t0, 1)
            return f" | {done}/{args.scenes} rendered ({rate * 3600:.0f}/hr)"
        except Exception:
            return ""

    _run(args, tasks, max_concurrent, check, t0)
