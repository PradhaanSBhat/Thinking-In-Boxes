# One-time asset index (models / HDRIs / materials) with relative paths.

import fcntl
import json
import os
import time

INDEX_FILENAME = ".asset_index_rel.json"

# Asset trees scanned under data_root, relative to it: the ARSG-110K bundles
# fetched by scripts/download_arsg_assets.sh.
HDRI_DIRS = ("ARSG-110K/hdrs",)
MATERIAL_DIRS = (
    "ARSG-110K/materials_floor",
    "ARSG-110K/materials_wall",
)
HDRI_SUFFIXES = (".hdr", ".exr")


def _relative_files(search_dir, data_root, suffixes, case_sensitive=False):
    """Every file under *search_dir* with one of *suffixes*, relative to data_root."""
    if not os.path.isdir(search_dir):
        return []
    found = []
    for root, _dirs, files in os.walk(search_dir):
        for name in files:
            candidate = name if case_sensitive else name.lower()
            if candidate.endswith(suffixes):
                found.append(os.path.relpath(os.path.join(root, name), data_root))
    return found


def _scan(data_root):
    """Walk the asset tree and build the index from scratch."""
    index = {"models": [], "hdris": [], "materials": []}

    # Models: any top-level dataset directory with a converted/ subdirectory.
    for entry in sorted(os.listdir(data_root)):
        index["models"] += _relative_files(
            os.path.join(data_root, entry, "converted"),
            data_root,
            (".blend",),
            case_sensitive=True,
        )
    print(f"  Models: {len(index['models'])}")

    for relative_dir in HDRI_DIRS:
        index["hdris"] += _relative_files(
            os.path.join(data_root, relative_dir), data_root, HDRI_SUFFIXES
        )
    print(f"  HDRIs: {len(index['hdris'])}")

    for relative_dir in MATERIAL_DIRS:
        index["materials"] += _relative_files(
            os.path.join(data_root, relative_dir), data_root, (".blend",)
        )
    print(f"  Materials: {len(index['materials'])}")
    return index


def _save_index(index, index_path):
    tmp = index_path + f".tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(index, f)
        os.replace(tmp, index_path)
    except Exception as e:
        print(f"  [WARN] Save failed: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def build_asset_index(data_root):
    """Load the cached asset index, building and caching it on first use."""
    index_path = os.path.join(data_root, INDEX_FILENAME)

    if os.path.isfile(index_path):
        t0 = time.time()
        with open(index_path) as f:
            index = json.load(f)
        print(
            f"[INDEX] Loaded in {time.time() - t0:.2f}s: "
            f"{len(index['models'])} models, {len(index['hdris'])} HDRIs, "
            f"{len(index['materials'])} materials"
        )
        return index

    print("[INDEX] Building asset index (one-time cost) ...")
    t0 = time.time()

    # File lock, so concurrent workers build it once.
    lock_fd = None
    try:
        lock_fd = open(index_path + ".lock", "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if os.path.isfile(index_path):
            with open(index_path) as f:
                index = json.load(f)
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            return index
    except Exception:
        pass

    index = _scan(data_root)
    print(f"  Built in {time.time() - t0:.1f}s")
    _save_index(index, index_path)

    if lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
        except Exception:
            pass

    return index
