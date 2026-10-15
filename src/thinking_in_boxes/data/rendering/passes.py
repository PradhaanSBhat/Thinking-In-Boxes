# The two render passes: a full render, and a masks-only fill-in.

import json
import os

import blenderproc as bproc
import bpy
import numpy as np

from blender.renderer import render_bbox_pass, render_depth_pass
from config.quality import OUTPUT_RESOLUTION
from rendering.bbox import draw_bboxes_and_masks
from rendering.images import to_depth_mm, to_mask8, to_rgb8
from rendering.masks import render_object_masks
from rendering.scene import (
    collect_object_metadata,
    pose_objects_for_view,
    prepare_bbox_scene,
    restore_main_scene,
    setup_scene,
    teardown_scene,
)

# ============================================================
# SHARED OUTPUT WRITERS
# ============================================================


def _write_face_masks(writer, scene_dir, vi, obj_masks, output_res):
    """mask_{v}_{o}.png — the bbox-face masks from the overlay pass."""
    for oi, mask in enumerate(obj_masks):
        writer.write_image(
            os.path.join(scene_dir, f"mask_{vi}_{oi}.png"), to_mask8(mask, output_res)
        )


def _write_bbox_coords(writer, scene_dir, vi, bbox_entries, output_res):
    """bbox_coords_{v}.json — tight 2D boxes from the full silhouettes."""
    writer.write_json(
        os.path.join(scene_dir, f"bbox_coords_{vi}.json"),
        {
            "view_id": vi,
            "image_width": output_res,
            "image_height": output_res,
            "objects": [e for e in bbox_entries if e is not None],
        },
    )


def _view_json(K, pose, vi, output_res, objects):
    return {
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "w": output_res,
        "h": output_res,
        "frame": {"file_path": f"rgb_{vi}.png", "transform_matrix": pose.tolist()},
        "objects": objects,
    }


def _attach_mask_filenames(objects_meta, placed_objects, vi):
    """Record each object's mask filenames in its view-JSON entry.

    objects_meta is indexed by *valid* object, placed_objects by slot, so the
    obj_mask name uses the slot index and the bbox_mask name the valid index.
    """
    valid_idx = 0
    for oi, obj in enumerate(placed_objects):
        if obj is None:
            continue
        if valid_idx < len(objects_meta):
            objects_meta[valid_idx]["obj_mask_file"] = f"obj_mask_{vi}_{oi}.png"
            objects_meta[valid_idx]["bbox_mask_file"] = f"mask_{vi}_{valid_idx}.png"
        valid_idx += 1
    return objects_meta


def _live_object_metadata(valid_objects, profile):
    return collect_object_metadata(
        valid_objects,
        record_bbox_3d=profile["record_bbox_3d"],
        record_objaverse_names=profile["record_objaverse_names"],
    )


# ============================================================
# FULL RENDER
# ============================================================


def render_scene(
    scene_dir,
    data_root,
    quality,
    writer,
    profile,
    emit_masks,
    output_res=OUTPUT_RESOLUTION,
):
    """Render a single scene from its metadata.json. Returns True on success.

    Bbox renders are done per-view: each view's bbox pass reuses the same
    camera pose that was just used for the RGB render, so the checkerboard
    floor shifts correctly with the camera perspective.
    """
    setup = setup_scene(scene_dir, data_root, quality, profile)
    if setup is None:
        return False
    meta, floor, bbox_strips, placed_objects, valid_objects, K, profile = setup

    force_xyz = profile["force_xyz_rotation_mode"]

    for view in meta["views"]:
        vi = view["view_id"]
        pose = np.array(view["camera_pose"])

        pose_objects_for_view(placed_objects, view, force_xyz)

        # --- RGB + depth render ---
        bproc.camera.add_camera_pose(pose)
        data = bproc.renderer.render()
        img, depth = data["colors"][0], data["depth"][0]

        # --- Bbox pass (same camera pose still active) ---
        # Hide objects, show checkerboard strips, switch to uniform lighting
        prepare_bbox_scene(valid_objects, floor, bbox_strips)
        floor_img = render_bbox_pass(quality["bbox_samples"])["colors"][0]
        restore_main_scene(valid_objects, floor, bbox_strips, data_root, meta)

        # Clear camera keyframes for next view
        bpy.context.scene.camera.animation_data_clear()

        # --- Post-processing + save ---
        bbox_img, obj_masks = draw_bboxes_and_masks(
            to_rgb8(floor_img, output_res), valid_objects, pose
        )
        writer.write_image(
            os.path.join(scene_dir, f"rgb_{vi}.png"), to_rgb8(img, output_res)
        )
        writer.write_image(os.path.join(scene_dir, f"bbox_{vi}.png"), bbox_img)

        # Depth and every mask file belong to the mask outputs, so they are
        # written only when masks are enabled.  The default render emits just
        # rgb / bbox / view JSON.
        if emit_masks:
            writer.write_image(
                os.path.join(scene_dir, f"depth_{vi}.png"),
                to_depth_mm(depth, output_res),
            )
            _write_face_masks(writer, scene_dir, vi, obj_masks, output_res)

            bbox_entries = render_object_masks(
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
                depth,
            )
            _write_bbox_coords(writer, scene_dir, vi, bbox_entries, output_res)

        obj_meta_live = _live_object_metadata(valid_objects, profile)
        if emit_masks:
            _attach_mask_filenames(obj_meta_live, placed_objects, vi)

        writer.write_json(
            os.path.join(scene_dir, f"view_{vi}.json"),
            _view_json(K, pose, vi, output_res, obj_meta_live),
        )

    teardown_scene(placed_objects, bbox_strips)
    return True


# ============================================================
# MASKS-ONLY RENDER
# ============================================================


def _missing_prerequisites(scene_dir, views):
    """True if any view lacks the RGB / bbox renders this pass fills in."""
    return any(
        not os.path.isfile(os.path.join(scene_dir, name))
        for view in views
        for name in (f"rgb_{view['view_id']}.png", f"bbox_{view['view_id']}.png")
    )


def _reload_view_json(writer, view_json_path):
    """Read back an existing view JSON, or None if absent or unreadable."""
    writer.flush()  # ensure any pending write to this file has landed
    if not os.path.isfile(view_json_path):
        return None
    try:
        with open(view_json_path) as f:
            return json.load(f)
    except Exception:
        return None


def render_scene_masks_only(
    scene_dir, data_root, quality, writer, profile, output_res=OUTPUT_RESOLUTION
):
    """Regenerate ONLY the pixel-accurate per-object masks for a scene.

    Assumes rgb_*.png / bbox_*.png already exist.  The RGB and bbox passes are
    skipped entirely; the full-scene depth needed for the occlusion test is
    rendered here at 1 sample rather than read back from disk.

    Returns True on success, False if the scene is unusable, and the string
    "missing" if the prerequisite renders are absent.
    """
    json_path = os.path.join(scene_dir, "metadata.json")
    if not os.path.isfile(json_path):
        return False

    with open(json_path) as f:
        probe_meta = json.load(f)

    # Verify prerequisites before doing any expensive setup.  Only the RGB and
    # bbox passes are required: the depth map is rendered here rather than
    # read back, since the default render no longer writes depth_*.png.
    if _missing_prerequisites(scene_dir, probe_meta["views"]):
        return "missing"

    setup = setup_scene(scene_dir, data_root, quality, profile)
    if setup is None:
        return False
    meta, floor, bbox_strips, placed_objects, valid_objects, K, profile = setup

    force_xyz = profile["force_xyz_rotation_mode"]

    for view in meta["views"]:
        vi = view["view_id"]
        pose = np.array(view["camera_pose"])

        pose_objects_for_view(placed_objects, view, force_xyz)

        # Full-scene depth for the occlusion test.  Rendered at 1 sample with
        # denoising off — only the depth pass is used, so shading quality is
        # irrelevant and this stays cheap.
        bproc.camera.add_camera_pose(pose)
        main_depth = render_depth_pass()
        bpy.context.scene.camera.animation_data_clear()

        writer.write_image(
            os.path.join(scene_dir, f"depth_{vi}.png"),
            to_depth_mm(main_depth, output_res),
        )

        # Bbox-face masks, derived from the same projected geometry the bbox
        # overlay uses.  draw_bboxes_and_masks needs an image to draw on; the
        # overlay itself is discarded here since bbox_{v}.png already exists.
        canvas = np.zeros((output_res, output_res, 3), dtype=np.uint8)
        _, obj_masks = draw_bboxes_and_masks(canvas, valid_objects, pose)
        _write_face_masks(writer, scene_dir, vi, obj_masks, output_res)

        bbox_entries = render_object_masks(
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
        )
        _write_bbox_coords(writer, scene_dir, vi, bbox_entries, output_res)

        # Refresh view_{v}.json so the mask filenames are recorded.  The
        # default render omits them, so they are added on this pass; the
        # camera and object metadata are rewritten identically.
        view_json_path = os.path.join(scene_dir, f"view_{vi}.json")
        existing = _reload_view_json(writer, view_json_path)
        if existing is None:
            existing = _view_json(
                K, pose, vi, output_res, _live_object_metadata(valid_objects, profile)
            )

        existing["objects"] = _attach_mask_filenames(
            existing.get("objects", []), placed_objects, vi
        )
        writer.write_json(view_json_path, existing)

        bpy.context.scene.camera.animation_data_clear()

    teardown_scene(placed_objects, bbox_strips)
    return True


# ============================================================
# RESTART PROBES
# ============================================================


def scene_is_rendered(scene_dir):
    """A scene counts as rendered once its first RGB view exists."""
    return os.path.isfile(os.path.join(scene_dir, "rgb_0.png"))


def scene_has_masks(scene_dir):
    """True if this scene already has per-object masks for every view.

    Used to make the masks-only pass restartable: a scene whose masks are
    already on disk is skipped rather than re-rendered.
    """
    meta_path = os.path.join(scene_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path) as f:
            views = json.load(f).get("views", [])
        present = os.listdir(scene_dir)
    except Exception:
        return False

    return bool(views) and all(
        any(f.startswith(f"obj_mask_{view.get('view_id', 0)}_") for f in present)
        for view in views
    )
