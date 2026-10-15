# Per-object metadata recorded into metadata.json.

import blenderproc as bproc
import bpy
import numpy as np
from mathutils import Matrix, Vector


def compute_object_mask_metadata(obj, obj_id, cam_pose, w, h):
    """Per-object segmentation mask metadata for a given view.

    Returns the object's projected 2D bounding box (for the object-only
    mask) and the category_id used during rendering to produce instance
    segmentation masks.
    """
    bpy.context.scene.camera.matrix_world = Matrix(cam_pose)
    bo = obj.blender_obj

    corners = bproc.camera.project_points(
        np.array([bo.matrix_world @ Vector(v) for v in bo.bound_box])
    )
    xs, ys = corners[:, 0], corners[:, 1]

    return {
        "object_id": obj_id,
        "category_id": obj_id + 1,  # 0 is background
        "bbox_2d": [
            max(0, int(np.floor(xs.min()))),  # clamped to the image bounds
            max(0, int(np.floor(ys.min()))),
            min(w, int(np.ceil(xs.max()))),
            min(h, int(np.ceil(ys.max()))),
        ],
        "bbox_2d_format": "xyxy",
        "mask_filename": f"mask_obj_{obj_id:02d}.png",  # expected render filename
    }
