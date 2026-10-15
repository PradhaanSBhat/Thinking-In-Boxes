# The `metadata.csv` bookkeeping the Objaverse-XL download needs.
#
# Derived from 3D-Fixer's ``dataset_toolkits/build_metadata.py`` (itself from
# TRELLIS) — see preparation/THIRD_PARTY_NOTICES.md.  Only two of that script's
# jobs matter here:
#
#   * build ``metadata.csv`` from the ARSG-110K asset listing, and
#   * merge the per-rank ``downloaded_*.csv`` records back into it, so a later
#     run knows what is already on disk and skips it.
#
# The upstream version also tracked renders, voxels, latents, features and
# aesthetic scores, none of which this pipeline produces.  Its statistics block
# additionally reads a ``captions`` column that the ARSG-110K listing does not
# necessarily have; the version here reports only columns it created itself.

import os
import shutil
import time

# The ARSG-110K listings that say which Objaverse-XL objects to fetch.
SOURCE_CSV = {
    "sketchfab": "hf://datasets/HorizonRobotics/ARSG-110K/ObjaverseXL_sketchfab.csv",
    "github": "hf://datasets/HorizonRobotics/ARSG-110K/ObjaverseXL_github.csv",
}

METADATA_CSV = "metadata.csv"
RECORDS_DIR = "merged_records"
REQUIRED_COLUMNS = ("sha256", "file_identifier")


def _read_csv(path):
    import pandas as pd

    return pd.read_csv(path)


def fetch_listing(source):
    """Download the ARSG-110K listing for *source* as a DataFrame."""
    if source not in SOURCE_CSV:
        raise ValueError(
            f"unknown source '{source}' (expected one of {sorted(SOURCE_CSV)})"
        )
    print(f"[METADATA] Fetching the {source} listing from ARSG-110K ...")
    listing = _read_csv(SOURCE_CSV[source])

    missing = [c for c in REQUIRED_COLUMNS if c not in listing.columns]
    if missing:
        raise ValueError(
            f"the ARSG-110K {source} listing is missing the column(s) "
            f"{missing}; it has {list(listing.columns)}.  The downloader needs "
            f"'sha256' to identify objects and 'file_identifier' to fetch them."
        )
    return listing


def load_or_build(output_dir, source):
    """Return the working metadata table, creating it on the first run.

    The listing is also kept verbatim as ``ObjaverseXL_<source>.csv`` so the
    original can be inspected after metadata.csv starts accumulating state.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, METADATA_CSV)

    if os.path.isfile(metadata_path):
        metadata = _read_csv(metadata_path)
        print(f"[METADATA] Loaded {len(metadata)} rows from {metadata_path}")
        return metadata

    metadata = fetch_listing(source)
    metadata.to_csv(os.path.join(output_dir, f"ObjaverseXL_{source}.csv"), index=False)
    metadata.to_csv(metadata_path, index=False)
    print(f"[METADATA] Wrote {len(metadata)} rows to {metadata_path}")
    return metadata


def merge_downloaded(output_dir):
    """Fold every ``downloaded_*.csv`` shard record into metadata.csv.

    Returns the merged table.  Shard files are moved into ``merged_records/``
    under a timestamp rather than deleted, matching the upstream behaviour.
    """
    import pandas as pd

    metadata_path = os.path.join(output_dir, METADATA_CSV)
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found — nothing to merge into. "
            f"Run the download first."
        )
    metadata = _read_csv(metadata_path).set_index("sha256")

    shard_files = sorted(
        f
        for f in os.listdir(output_dir)
        if f.startswith("downloaded_") and f.endswith(".csv")
    )
    parts = []
    for name in shard_files:
        try:
            parts.append(_read_csv(os.path.join(output_dir, name)))
        except Exception as e:  # a worker killed mid-write leaves a partial file
            print(f"  [WARN] skipping unreadable {name}: {e}")

    if parts:
        downloaded = pd.concat(parts).set_index("sha256")
        # Keep the newest record for a sha256 that several shards reported.
        downloaded = downloaded[~downloaded.index.duplicated(keep="last")]
        if "local_path" in metadata.columns:
            metadata.update(downloaded, overwrite=True)
        else:
            metadata = metadata.join(downloaded, on="sha256", how="left")

        archive = os.path.join(output_dir, RECORDS_DIR)
        os.makedirs(archive, exist_ok=True)
        stamp = str(int(time.time()))
        for name in shard_files:
            shutil.move(
                os.path.join(output_dir, name), os.path.join(archive, f"{stamp}_{name}")
            )

    metadata.to_csv(metadata_path)
    return metadata.reset_index()


def write_statistics(output_dir, metadata):
    """Write (and print) statistics.txt for the columns we actually maintain."""
    total = len(metadata)
    downloaded = (
        int(metadata["local_path"].count()) if "local_path" in metadata.columns else 0
    )
    text = (
        "Statistics:\n"
        f"  - Number of assets: {total}\n"
        f"  - Number of assets downloaded: {downloaded}\n"
        f"  - Remaining: {total - downloaded}\n"
    )
    with open(os.path.join(output_dir, "statistics.txt"), "w") as f:
        f.write(text)
    print(text, end="")
    return downloaded, total
