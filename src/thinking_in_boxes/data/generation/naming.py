# Objaverse UID / human-readable name lookup (lazy, optional dependency).

import os

_objaverse_annotations_cache = {}


def extract_uid_from_path(model_rel):
    """Extract objaverse UID from model path.

    'ObjaverseXL_sketchfab/converted/.../004fb1048506417287f7590b053d7e18.blend'
    → '004fb1048506417287f7590b053d7e18'
    """
    basename = os.path.basename(model_rel)
    uid = basename.split(".")[0]
    return uid if uid else None


def get_objaverse_name(uid):
    """Get the human-readable name for an objaverse UID."""
    if uid is None:
        return None
    if uid in _objaverse_annotations_cache:
        return _objaverse_annotations_cache[uid]
    try:
        import objaverse

        annotations = objaverse.load_annotations(uids=[uid])
        name = annotations.get(uid, {}).get("name", None)
        _objaverse_annotations_cache[uid] = name
        return name
    except Exception as e:
        print(f"  [WARN] objaverse lookup failed for {uid}: {e}")
        _objaverse_annotations_cache[uid] = None
        return None
