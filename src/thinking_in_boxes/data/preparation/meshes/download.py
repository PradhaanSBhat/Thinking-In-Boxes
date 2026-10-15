# Sharded Objaverse-XL download.
#
# Derived from 3D-Fixer's ``dataset_toolkits/download.py`` and
# ``dataset_toolkits/datasets/ObjaverseXL.py`` — see
# preparation/THIRD_PARTY_NOTICES.md.
#
# Two deliberate differences from upstream:
#
#   * the sketchfab source calls ``SketchfabDownloader`` directly instead of
#     ``objaverse.xl.download_objects``.  The latter dispatches on a ``source``
#     column and, to build the annotation table it dispatches from, downloads the
#     annotations for GitHub, Thingiverse and Smithsonian as well — gigabytes of
#     parquet for a sketchfab-only run.
#   * ``--world-size``/``--rank`` are plain arguments rather than a fixed 1/16
#     slice baked into a shell script.
#
# The ARSG-110K sketchfab listing has these columns (verified against the live
# file), and no ``local_path`` until the first merge adds one::
#
#     sha256,file_identifier,aesthetic_score,captions
#
# It holds roughly 163,000 objects — the TRELLIS-500K ObjaverseXL (sketchfab)
# subset — so the default 1/16 shard is about 10,200 of them.
#
# Resuming is handled at two levels: rows that already have a ``local_path`` in
# metadata.csv are filtered out here, and the Objaverse downloader itself skips
# any .glb already present on disk.

import os

RAW_SUBDIR = "raw"

# The full listing is ~163k objects; the original shell script hard-coded
# `--world_size 16 --rank 0`, i.e. ~10.2k.  Kept as the default so a plain run
# fetches the same slice it always did rather than 16x more.
DEFAULT_WORLD_SIZE = 16


def shard(metadata, rank, world_size):
    """The ``rank``-th contiguous slice of *metadata* out of *world_size*."""
    if world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError(f"--rank must be in [0, {world_size})")
    start = len(metadata) * rank // world_size
    end = len(metadata) * (rank + 1) // world_size
    return metadata[start:end]


def pending(metadata):
    """Rows not yet downloaded (no local_path recorded)."""
    if "local_path" not in metadata.columns:
        return metadata
    return metadata[metadata["local_path"].isna()]


def filter_aesthetic(metadata, minimum):
    """Drop rows below an aesthetic score, using the listing's own column.

    Objaverse is uneven; the listing carries TRELLIS's aesthetic_score, so a
    threshold is the cheapest way to bias the download toward usable models.
    """
    if minimum is None or "aesthetic_score" not in metadata.columns:
        return metadata
    return metadata[metadata["aesthetic_score"] >= minimum]


def limited(metadata, limit):
    """The first *limit* rows, or all of them when limit is None."""
    return metadata if not limit else metadata[:limit]


def download(metadata, output_dir, source="sketchfab", processes=None):
    """Download the objects in *metadata*; return a sha256 -> local_path frame.

    Files land under ``<output_dir>/raw/hf-objaverse-v1/glbs/<prefix>/<uid>.glb``,
    which is where the GLB converter expects to find them.
    """
    import pandas as pd

    raw_dir = os.path.join(output_dir, RAW_SUBDIR)
    os.makedirs(raw_dir, exist_ok=True)

    if source == "sketchfab":
        from objaverse.xl.sketchfab import SketchfabDownloader

        annotations = SketchfabDownloader.get_annotations()
        annotations = annotations[annotations["sha256"].isin(metadata["sha256"].values)]
        file_paths = SketchfabDownloader.download_objects(
            annotations, download_dir=raw_dir, processes=processes
        )
    else:
        import objaverse.xl as oxl

        annotations = oxl.get_annotations()
        annotations = annotations[annotations["sha256"].isin(metadata["sha256"].values)]
        file_paths = oxl.download_objects(
            annotations,
            download_dir=raw_dir,
            processes=processes,
            save_repo_format="zip",
        )

    # A duplicated file_identifier would make .loc return a Series, so build a
    # plain first-wins mapping instead of indexing the frame.
    sha_by_identifier = dict(zip(metadata["file_identifier"], metadata["sha256"]))
    downloaded = {}
    for identifier, path in file_paths.items():
        sha256 = sha_by_identifier.get(identifier)
        if sha256 is None:
            # The annotation table is Objaverse-wide; a path we never asked
            # for can come back, and it has no row here to attribute it to.
            continue
        downloaded[sha256] = os.path.relpath(path, output_dir)

    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])


def write_shard_record(frame, output_dir, rank):
    """Persist one shard's results as downloaded_<rank>.csv."""
    path = os.path.join(output_dir, f"downloaded_{rank}.csv")
    frame.to_csv(path, index=False)
    print(f"[DOWNLOAD] Recorded {len(frame)} objects in {path}")
    return path
