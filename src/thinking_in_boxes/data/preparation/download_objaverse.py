#!/usr/bin/env python3
# Download the Objaverse-XL models listed by ARSG-110K.
#
# Replaces the "clone 3D-Fixer and run its dataset_toolkits" step: the small
# part of that toolkit this pipeline needs is vendored under
# ``preparation/meshes`` (see preparation/THIRD_PARTY_NOTICES.md).
#
# By default the models land directly in the configured asset root, at
# ``<data_root>/<assets.dataset>/raw/hf-objaverse-v1/glbs/``, which is exactly
# where ``scripts/convert_assets.sh`` looks — nothing to move or symlink
# afterwards.
#
# The ARSG-110K sketchfab listing holds ~163,000 objects. The default is the
# 1/16 shard the original shell script hard-coded, ~10,200 objects; pass
# ``--world-size 1`` for the lot.
#
# Usage:
#   ./scripts/download_objaverse.sh                      # ~10k objects (1/16)
#   ./scripts/download_objaverse.sh --world-size 1       # all ~163k
#   ./scripts/download_objaverse.sh --limit 500          # a quick sample
#   ./scripts/download_objaverse.sh --world-size 16 --rank 3   # a different slice
#   ./scripts/download_objaverse.sh --min-aesthetic-score 5.5  # better models only
#
# Long transfers drop, so the download is retried until it completes (no cap
# unless --retries N is given); each attempt skips whatever is already on disk.
# Shard the work across machines by giving each one a different --rank with the
# same --world-size.

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.paths import abspath  # noqa: E402
from config.settings import CONFIG_FILENAME, load_settings  # noqa: E402
from preparation.meshes import download as dl  # noqa: E402
from preparation.meshes import metadata as md  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download the Objaverse-XL models listed by ARSG-110K. "
        f"Defaults come from {CONFIG_FILENAME}."
    )
    parser.add_argument("--config", default=None, metavar="PATH")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where the dataset is assembled "
        "(default: <paths.data_root>/<assets.dataset>)",
    )
    parser.add_argument(
        "--source",
        choices=sorted(md.SOURCE_CSV),
        default="sketchfab",
        help="Which ARSG-110K listing to download (default: sketchfab)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=dl.DEFAULT_WORLD_SIZE,
        help=f"Split the listing into this many shards (default: "
        f"{dl.DEFAULT_WORLD_SIZE}, i.e. ~10k of the ~163k objects). Pass 1 for "
        f"the whole listing.",
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Which shard to download (default: 0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many objects, applied after sharding",
    )
    parser.add_argument(
        "--min-aesthetic-score",
        type=float,
        default=None,
        help="Skip objects below this aesthetic score (the listing carries one)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Parallel downloads (default: one per CPU)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Attempts before giving up (default: 0, i.e. retry until it "
        "completes -- connection resets can take many tries)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Seconds between attempts (default: 10)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Build/refresh metadata.csv and exit without downloading",
    )
    return parser


def resolve_output_dir(args):
    if args.output_dir:
        return abspath(args.output_dir)
    cfg = load_settings(args.config)
    return os.path.join(cfg.require_data_root(), cfg.assets["dataset"])


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        output_dir = resolve_output_dir(args)
    except ValueError as e:
        parser.error(f"{e}\n  ...or pass --output-dir to this script directly.")

    print(f"[OBJAVERSE] Source: {args.source}")
    print(f"[OBJAVERSE] Output: {output_dir}")

    try:
        metadata = md.load_or_build(output_dir, args.source)
    except Exception as e:
        print(f"[ERROR] could not build metadata: {e}")
        sys.exit(1)

    if args.metadata_only:
        md.write_statistics(output_dir, metadata)
        return

    eligible = dl.filter_aesthetic(dl.pending(metadata), args.min_aesthetic_score)
    try:
        todo = dl.limited(dl.shard(eligible, args.rank, args.world_size), args.limit)
    except ValueError as e:
        parser.error(str(e))

    print(
        f"[OBJAVERSE] Listing: {len(metadata)} objects, "
        f"{len(eligible)} still to fetch"
        + (
            f" (aesthetic >= {args.min_aesthetic_score})"
            if args.min_aesthetic_score
            else ""
        )
    )
    print(
        f"[OBJAVERSE] Shard {args.rank}/{args.world_size}"
        + (f", limit {args.limit}" if args.limit else "")
        + f": {len(todo)} objects this run"
    )
    if todo.empty:
        print("[OBJAVERSE] Nothing to do.")
        md.write_statistics(output_dir, metadata)
        return

    max_attempts = args.retries if args.retries > 0 else None
    for attempt in itertools.count(1):
        print(f"\n[OBJAVERSE] Attempt {attempt}/{max_attempts or 'unlimited'} ...")
        try:
            frame = dl.download(todo, output_dir, args.source, args.processes)
            dl.write_shard_record(frame, output_dir, args.rank)
            break
        except KeyboardInterrupt:
            print("\n[OBJAVERSE] Interrupted.")
            sys.exit(130)
        except Exception as e:
            print(f"[WARN] attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt == max_attempts:
                print("[ERROR] giving up; re-run to continue where it stopped.")
                sys.exit(1)
            time.sleep(args.retry_delay)

    metadata = md.merge_downloaded(output_dir)
    md.write_statistics(output_dir, metadata)
    print(
        f"\n[OBJAVERSE] GLBs are under "
        f"{os.path.join(output_dir, dl.RAW_SUBDIR, 'hf-objaverse-v1', 'glbs')}\n"
        f"            Next: ./scripts/convert_assets.sh"
    )


if __name__ == "__main__":
    main()
