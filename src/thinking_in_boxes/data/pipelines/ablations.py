# Single-factor ablation runner.
#
# Builds the ablation datasets declared under `ablations:` in config.yaml.
#
# Each entry isolates one kind of object variation, so a model's sensitivity to
# translation / scaling / rotation can be measured independently.  All of them
# use the test split in object_movement mode; only the enabled factors differ::
#
#     ablations:
#       only_rotate:
#         scenes: 100
#         translation: false
#         scaling: false
#         rotation: true

import argparse
import os
import subprocess
import sys

from config.project import DISTRIBUTE_SCRIPT, LAUNCH_SCRIPT
from config.settings import CONFIG_FILENAME, load_settings


def build_commands(name, spec, cfg, config_path, skip_render):
    """Return the (generate, render) commands for one ablation."""
    scenes = int(spec.get("scenes", 100))
    out_dir = os.path.join(cfg.paths["ablation_root"], name)

    def factor(key):
        # true -> --enable-X, false -> --no-X
        return "enable" if spec.get(key) else "no"

    generate = [
        sys.executable,
        DISTRIBUTE_SCRIPT,
        "--split",
        "test",
        "--mode",
        "object_movement",
        "--output-dir",
        out_dir,
        "--global-start",
        "0",
        "--global-count",
        str(scenes),
        "--log-dir",
        os.path.join(cfg.paths["log_root"], f"generate_{name}"),
        f"--generator-flag=--{factor('translation')}-translation",
        f"--generator-flag=--{factor('scaling')}-scaling",
        f"--generator-flag=--{factor('rotation')}-rotation",
    ]

    render = None
    if not skip_render:
        render = [
            sys.executable,
            LAUNCH_SCRIPT,
            "render",
            "--split",
            "test",
            "--scene-dir",
            out_dir,
            "--start",
            "0",
            "--scenes",
            str(scenes),
            "--log-dir",
            os.path.join(cfg.paths["log_root"], f"render_{name}"),
        ]

    if config_path:
        generate[2:2] = ["--config", config_path]
        if render:
            render[3:3] = ["--config", config_path]

    return generate, render


def main():
    ap = argparse.ArgumentParser(
        description="Run the ablation datasets declared in config.yaml."
    )
    ap.add_argument(
        "names", nargs="*", help="Ablations to run (default: all in the config)"
    )
    ap.add_argument(
        "--config",
        default=None,
        help=f"Configuration file (default: {CONFIG_FILENAME})",
    )
    ap.add_argument(
        "--skip-render",
        action="store_true",
        help="Generate scene metadata but do not render",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print the commands without running them"
    )
    args = ap.parse_args()

    try:
        cfg = load_settings(args.config)
        cfg.require_data_root()
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    table = cfg.ablations or {}
    if not table:
        print(
            f"[ERROR] no `ablations:` section in {cfg.source_path or CONFIG_FILENAME}"
        )
        sys.exit(1)

    selected = args.names or list(table)
    unknown = [n for n in selected if n not in table]
    if unknown:
        print(f"[ERROR] unknown ablation(s): {', '.join(unknown)}")
        print(f"        available: {', '.join(table)}")
        sys.exit(1)

    print(f"[ABLATIONS] {len(selected)} to run: {', '.join(selected)}")

    for name in selected:
        spec = table[name] or {}
        generate, render = build_commands(
            name, spec, cfg, args.config, args.skip_render
        )

        factors = [k for k in ("translation", "scaling", "rotation") if spec.get(k)]
        print(
            f"\n=== {name} "
            f"({spec.get('scenes', 100)} scenes, "
            f"{'+'.join(factors) if factors else 'no factors'}) ==="
        )

        for cmd in (generate, render):
            if cmd is None:
                continue
            if args.dry_run:
                print("  " + " ".join(cmd))
                continue
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[ERROR] {name}: command failed (exit {result.returncode})")
                sys.exit(result.returncode)

    print("\n[DONE]")
