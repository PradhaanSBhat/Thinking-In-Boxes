# Pixel-accurate per-object masks, via solo renders and a depth test.

import os

import blenderproc as bproc
import bpy
import cv2
import numpy as np

from blender.objects import set_render_visibility
from blender.renderer import temporary_render_settings
from blender.world import setup_lighting_from_meta
from rendering.images import resized, to_mask8

DEPTH_TOL = 0.1  # 10 cm absolute tolerance for the occlusion test

EMPTY_BBOX = {
    "bbox_2d": [0, 0, 0, 0],
    "bbox_2d_format": "xyxy",
    "center_2d": [0, 0],
    "width": 0,
    "height": 0,
    "pixel_count": 0,
}


def _bbox_entry(object_id, mask):
    """Tight 2D box of a silhouette, or a zeroed entry if it is empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"object_id": object_id, **EMPTY_BBOX}
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return {
        "object_id": object_id,
        "bbox_2d": [x_min, y_min, x_max, y_max],
        "bbox_2d_format": "xyxy",
        "center_2d": [
            round((x_min + x_max) / 2.0, 2),
            round((y_min + y_max) / 2.0, 2),
        ],
        "width": x_max - x_min,
        "height": y_max - y_min,
        "pixel_count": int(len(xs)),
    }


def render_object_masks(
    placed_objects,
    floor,
    bbox_strips,
    pose,
    scene_dir,
    vi,
    writer,
    output_res,
    data_root,
    meta,
    main_depth,
):
    """Render pixel-accurate per-object masks using transparent film + depth.

    Each object is rendered solo (everything else hidden) with transparent
    film at 1 sample.  The resulting depth is compared against *main_depth*
    (the full-scene depth from the RGB render) to handle occlusion:

      - solo_depth ≈ main_depth  →  this object is the closest surface  →  visible
      - solo_depth > main_depth  →  another object was in front          →  occluded

    This mirrors the ``occupied`` buffer logic in draw_bboxes_and_masks.

    Returns:
        bbox_results: list aligned with placed_objects.  None for failed
        objects, otherwise a dict with the tight 2D bbox computed from the
        FULL (unoccluded) silhouette.
    """
    valid_objects = [o for o in placed_objects if o is not None]
    bbox_results = []

    with temporary_render_settings(
        film_transparent=True, samples=1, use_denoising=False
    ):
        # Hide everything that is not an object we want a mask for.
        floor.blender_obj.hide_render = True
        floor.blender_obj.cycles.is_shadow_catcher = False
        for strip in bbox_strips:
            strip.hide_render = True
        set_render_visibility(valid_objects, False)

        for oi, obj in enumerate(placed_objects):
            if obj is None:
                bbox_results.append(None)
                continue

            set_render_visibility([obj], True)
            bpy.context.view_layer.update()

            # The depth pass is already enabled globally.
            bproc.camera.add_camera_pose(pose)
            solo_depth = bproc.renderer.render()["depth"][0]
            if solo_depth.shape != main_depth.shape:
                solo_depth = resized(solo_depth, main_depth.shape[0], cv2.INTER_NEAREST)

            # Step 1: the full silhouette, ignoring occlusion — this is what
            # the 2D bbox is measured from.
            is_object = np.isfinite(solo_depth) & (solo_depth > 0) & (solo_depth < 1e8)
            full_mask = is_object.astype(np.uint8) * 255
            bbox_results.append(_bbox_entry(oi, to_mask8(full_mask, output_res)))

            # Step 2: drop the pixels another object is in front of, and save.
            is_visible = is_object & (solo_depth <= main_depth + DEPTH_TOL)
            writer.write_image(
                os.path.join(scene_dir, f"obj_mask_{vi}_{oi}.png"),
                to_mask8(is_visible.astype(np.uint8) * 255, output_res),
            )

            set_render_visibility([obj], False)
            # Clear camera keyframes so the next add_camera_pose works.
            bpy.context.scene.camera.animation_data_clear()

    set_render_visibility(valid_objects, True)
    floor.blender_obj.hide_render = False
    floor.blender_obj.cycles.is_shadow_catcher = True
    setup_lighting_from_meta(data_root, meta)  # re-establish HDRI lighting

    return bbox_results
