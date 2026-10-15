# Argument parsing and settings resolution for launch.py.

import argparse
import os

from config.paths import abspath
from config.settings import CONFIG_FILENAME, load_settings
from launching.commands import cmd_generate, cmd_render


def main():
    parser = argparse.ArgumentParser(
        description="Distributed scene generation + rendering. "
        f"Defaults come from {CONFIG_FILENAME}; any argument "
        "given here overrides it."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=f"Configuration file (default: {CONFIG_FILENAME} next to this script)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- GENERATE ---
    sg = sub.add_parser("generate", help="Generate scene metadata")
    sg.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which dataset split profile to use (default: train)",
    )
    sg.add_argument(
        "--mode",
        choices=["mixed", "camera_movement", "object_movement"],
        default=None,
        help="Override the split's default movement mode",
    )
    sg.add_argument(
        "--data-root",
        default=None,
        help="Asset dataset root (default: paths.data_root)",
    )
    sg.add_argument(
        "--output-dir",
        default=None,
        help="Where scene metadata is written "
        "(default: the split's configured scene directory)",
    )
    sg.add_argument(
        "--scenes",
        type=int,
        default=None,
        help="How many scenes (default: scale.train_scenes / scale.test_scenes)",
    )
    sg.add_argument(
        "--start",
        type=int,
        default=None,
        help="First global scene index (default: scale.train_start)",
    )
    sg.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: scale.num_workers)",
    )
    sg.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Max scenes per blenderproc launch (default 500)",
    )
    sg.add_argument("--num-views", type=int, default=None)
    sg.add_argument(
        "--min-obj",
        type=int,
        default=None,
        help="Min objects per scene (default: objects.<split>_min)",
    )
    sg.add_argument(
        "--max-obj",
        type=int,
        default=None,
        help="Max objects per scene (default: objects.<split>_max)",
    )
    sg.add_argument(
        "--generator-flag",
        action="append",
        default=None,
        metavar="FLAG",
        help="Extra flag passed through to generate_scenes.py "
        "(repeatable). Use the '=' form so argparse does not "
        "mistake the value for an option of its own, e.g. "
        "--generator-flag=--no-scaling",
    )
    sg.add_argument("--script", default=None)
    sg.add_argument(
        "--log-dir", default=None, help="Worker logs (default: <log_root>/generate)"
    )
    sg.add_argument("--stagger", type=float, default=2.0)
    sg.add_argument("--monitor-sec", type=float, default=15.0)
    sg.add_argument("--dry-run", action="store_true")

    # --- RENDER ---
    sr = sub.add_parser("render", help="Render from metadata")
    sr.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which dataset split profile to use (default: train)",
    )
    sr.add_argument(
        "--mode",
        choices=["camera_movement", "object_movement"],
        default=None,
        help="Which test-split scene directory to render (default: camera_movement)",
    )
    sr.add_argument(
        "--data-root",
        default=None,
        help="Asset dataset root (default: paths.data_root)",
    )
    sr.add_argument(
        "--scene-dir",
        default=None,
        help="Directory holding the generated scenes "
        "(default: the split's configured scene directory)",
    )
    sr.add_argument(
        "--scenes",
        type=int,
        default=None,
        help="How many scenes (default: scale.train_scenes / scale.test_scenes)",
    )
    sr.add_argument(
        "--start",
        type=int,
        default=None,
        help="First scene index (default: scale.train_start)",
    )
    sr.add_argument(
        "--procs-per-gpu",
        type=int,
        default=None,
        help="Render processes per GPU (default: render.procs_per_gpu)",
    )
    sr.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Max scenes per blenderproc launch (default: render.batch_size)",
    )
    sr.add_argument("--gpus", type=int, default=0)
    sr.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids (default: render.gpu_ids)",
    )
    sr.add_argument(
        "--fast",
        action="store_true",
        default=None,
        help="Draft quality (default: render.fast)",
    )
    sr.add_argument(
        "--masks-only",
        action="store_true",
        help="Generate ONLY the per-object mask PNGs, assuming "
        "RGB and bbox images already exist. If they do not, "
        "you are asked whether to render everything.",
    )
    sr.add_argument(
        "--with-masks",
        action="store_true",
        help="Force per-object masks on during a full render",
    )
    sr.add_argument(
        "--no-masks",
        action="store_true",
        help="Force per-object masks off during a full render",
    )
    sr.add_argument(
        "--yes",
        action="store_true",
        help="Assume yes to the full-render fallback prompt",
    )
    sr.add_argument("--script", default=None)
    sr.add_argument(
        "--log-dir", default=None, help="Worker logs (default: <log_root>/render)"
    )
    sr.add_argument("--stagger", type=float, default=5.0)
    sr.add_argument("--monitor-sec", type=float, default=30.0)
    sr.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "render" and args.with_masks and args.no_masks:
        parser.error("--with-masks and --no-masks are mutually exclusive")

    # --- Resolve settings: CLI argument > config.yaml > built-in default ---
    try:
        cfg = load_settings(args.config)
        args.data_root = (
            abspath(args.data_root) if args.data_root else cfg.require_data_root()
        )
    except ValueError as e:
        parser.error(str(e))

    if args.scenes is None:
        args.scenes = cfg.scenes_for(args.split)
    if args.start is None:
        args.start = cfg.start_for(args.split)

    if args.command == "generate":
        if args.workers is None:
            args.workers = cfg.scale["num_workers"]
        if not args.output_dir:
            args.output_dir = cfg.scene_dir_for(args.split, args.mode)
        args.output_dir = abspath(args.output_dir)
        if not args.log_dir:
            args.log_dir = os.path.join(cfg.paths["log_root"], "generate")
        try:
            args.min_obj, args.max_obj = cfg.object_range(
                args.split, args.min_obj, args.max_obj
            )
        except ValueError as e:
            parser.error(str(e))
    else:
        if not args.scene_dir:
            args.scene_dir = cfg.scene_dir_for(args.split, args.mode)
        args.scene_dir = abspath(args.scene_dir)
        if not args.log_dir:
            args.log_dir = os.path.join(cfg.paths["log_root"], "render")
        if args.procs_per_gpu is None:
            args.procs_per_gpu = cfg.render["procs_per_gpu"]
        if args.batch_size is None:
            args.batch_size = cfg.render["batch_size"]
        if args.gpu_ids is None:
            args.gpu_ids = ",".join(str(g) for g in cfg.gpu_ids())
        if args.fast is None:
            args.fast = bool(cfg.render.get("fast"))
        # render.masks: true adds every mask output to a full render.
        # null/false leave the default (rgb + bbox + view JSON only).
        if not (args.with_masks or args.no_masks or args.masks_only):
            if cfg.render.get("masks") is True:
                args.with_masks = True

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "render":
        cmd_render(args)
