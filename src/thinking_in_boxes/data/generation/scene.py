# Composition of a single scene and its metadata.json.

import json
import os
import random

import blenderproc as bproc
import bpy
import numpy as np

from blender.objects import (
    align_object_to_floor,
    bbox_dims,
    delete_object,
    load_blend,
    normalize_object,
)
from config.quality import CAMERA_LENS_MM, RENDER_RESOLUTION
from generation.metadata import compute_object_mask_metadata
from generation.naming import extract_uid_from_path, get_objaverse_name
from generation.perturbation import perturb_mixed, perturb_object_movement
from generation.placement import compute_camera_poses, get_xy_bounds, try_place_object

MAX_PLACEMENT_ATTEMPTS = 15


def _place_objects(data_root, model_files, profile):
    """Load, normalise and place objects until the scene is full.

    Returns (objects, base_scales, all_coords, object_metadata).
    """
    force_xyz = profile["force_xyz_rotation_mode"]
    target_size = profile["normalize_target_size"]

    objects, base_scales, placed_infos, all_coords, metadata = [], [], [], [], []
    num_to_place = random.randint(profile["min_obj"], profile["max_obj"])

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        if len(objects) >= num_to_place:
            break

        model_rel = random.choice(model_files)
        obj = load_blend(
            os.path.join(data_root, model_rel), force_xyz, unhide=False, warn=False
        )
        if obj is None:
            continue

        base_scale = normalize_object(
            obj, target_size, force_xyz, profile["apply_rotation_on_normalize"]
        )
        obj.blender_obj.rotation_euler = (0, 0, random.uniform(0, 2 * np.pi))
        bpy.context.view_layer.update()

        placement = try_place_object(obj, placed_infos)
        if placement is None:
            delete_object(obj)
            continue
        placed_infos.append(placement)
        align_object_to_floor(obj)

        entry = {
            "object_id": len(objects),
            "model_path": model_rel,
            "source_file": os.path.basename(model_rel),
            "base_scale": base_scale,
            "original_location": obj.get_location().tolist(),
            "original_rotation": list(obj.blender_obj.rotation_euler),
            "original_scale": list(obj.blender_obj.scale),
        }
        if profile["record_objaverse_names"]:
            uid = extract_uid_from_path(model_rel)
            entry["objaverse_uid"] = uid
            entry["objaverse_name"] = get_objaverse_name(uid)
        if profile["record_bbox_3d"]:
            entry["bbox_3d"] = bbox_dims(obj)

        objects.append(obj)
        base_scales.append(base_scale)
        metadata.append(entry)
        all_coords.extend(obj.get_bound_box())

    return objects, base_scales, all_coords, metadata


def _camera_poses(all_coords, objects, num_views, camera_moves):
    """Poses per view: distinct ones when the camera moves, else one reused."""
    if camera_moves:
        return compute_camera_poses(all_coords, objects, num_views)
    poses = compute_camera_poses(all_coords, objects, 1)
    return [poses[0]] * num_views if poses else poses


def _reset_objects(objects, base_scales, original_locations, original_rotations):
    """Return every object to the state it was placed in."""
    for obj, base_scale in zip(objects, base_scales):
        if obj.blender_obj.animation_data:
            obj.blender_obj.animation_data_clear()
        obj.set_location(original_locations[obj.get_name()])
        obj.set_scale([base_scale] * 3)
        obj.blender_obj.rotation_euler = original_rotations[obj.get_name()]
    bpy.context.view_layer.update()
    for obj in objects:
        align_object_to_floor(obj)


def _object_states(objects, record_bbox_3d):
    states = []
    for oi, obj in enumerate(objects):
        state = {
            "object_id": oi,
            "location": obj.get_location().tolist(),
            "rotation_euler": list(obj.blender_obj.rotation_euler),
            "scale": list(obj.blender_obj.scale),
        }
        if record_bbox_3d:
            state["bbox_3d"] = bbox_dims(obj)
        states.append(state)
    return states


def _build_views(objects, base_scales, poses, all_coords, profile):
    """One entry per view, applying the mode's perturbation to views after 0."""
    mode = profile["mode"]
    coords = np.array(all_coords)
    scene_radius = np.max(np.linalg.norm(coords - np.mean(coords, axis=0), axis=1))
    render = bpy.context.scene.render
    w, h = render.resolution_x, render.resolution_y

    original_locations = {o.get_name(): o.get_location().copy() for o in objects}
    original_rotations = {
        o.get_name(): list(o.blender_obj.rotation_euler) for o in objects
    }

    views = []
    for vi, pose in enumerate(poses):
        _reset_objects(objects, base_scales, original_locations, original_rotations)
        bounds = [get_xy_bounds(o) for o in objects]

        # NOTE: in pure camera_movement mode this is skipped entirely — objects
        # remain at their original state for every view and only the camera
        # pose differs between views.
        if vi > 0:
            perturb = {
                "mixed": perturb_mixed,
                "object_movement": perturb_object_movement,
            }.get(mode)
            # v6 only perturbed when translation or scaling was enabled.
            if mode == "mixed" and not (
                profile["enable_translation"] or profile["enable_scaling"]
            ):
                perturb = None
            if perturb:
                perturb(objects, base_scales, bounds, pose, w, h, scene_radius, profile)

        bpy.context.view_layer.update()
        view_entry = {
            "view_id": vi,
            "camera_pose": pose.tolist(),
            "objects_state": _object_states(objects, profile["record_bbox_3d"]),
        }
        if profile["record_object_masks"]:
            view_entry["object_masks"] = [
                compute_object_mask_metadata(obj, oi, pose, w, h)
                for oi, obj in enumerate(objects)
            ]
        views.append(view_entry)

    return views


def generate_one_scene(scene_idx, data_root, index, num_views, output_dir, profile):
    """Build one scene and write its metadata.json.  True if it was written.

    profile["mode"] selects the per-view variation:
      - mixed           : camera moves AND a random object is perturbed (train)
      - camera_movement : camera changes per view, objects stay fixed
      - object_movement : objects change per view, camera stays fixed
    """
    mode = profile["mode"]
    camera_moves = mode in ("camera_movement", "mixed")
    only_object_movement = mode == "object_movement"

    bproc.clean_up()
    bproc.camera.set_resolution(RENDER_RESOLUTION, RENDER_RESOLUTION)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=CAMERA_LENS_MM, lens_unit="MILLIMETERS"
    )

    hdri = {
        "path": random.choice(index["hdris"]),
        "rotation_z": random.uniform(0, 6.28),
        "strength": random.uniform(0.8, 1.6),
    }
    materials = index["materials"]
    floor_texture = random.choice(materials) if materials else None

    objects, base_scales, all_coords, obj_meta = _place_objects(
        data_root, index["models"], profile
    )
    if not objects:
        return False

    # We always need at least one valid camera pose.
    poses = _camera_poses(all_coords, objects, num_views, camera_moves)
    if not poses:
        for obj in objects:
            delete_object(obj)
        return False

    K = bproc.camera.get_intrinsics_as_K_matrix()
    views = _build_views(objects, base_scales, poses, all_coords, profile)

    scene_data = {
        "scene_id": scene_idx,
        "num_objects": len(objects),
        "num_views": len(views),
        "split": profile["name"],
        "mode": mode,
        "camera_movement": camera_moves,
        "object_movement": only_object_movement,
        "normalize_target_size": profile["normalize_target_size"],
        # How objects were loaded; the renderer reads this back so that the
        # recorded rotation_euler is applied exactly as it was here.
        "object_loading": {
            "force_xyz_rotation_mode": profile["force_xyz_rotation_mode"],
            "apply_rotation_on_normalize": profile["apply_rotation_on_normalize"],
            "normalize_target_size": profile["normalize_target_size"],
        },
        "camera": {
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "w": RENDER_RESOLUTION,
            "h": RENDER_RESOLUTION,
            "lens_mm": CAMERA_LENS_MM,
        },
        "hdri": hdri,
        "floor_texture": floor_texture,
        "objects": obj_meta,
        "views": views,
    }

    scene_dir = os.path.join(output_dir, f"scene_{scene_idx:06d}")
    os.makedirs(scene_dir, exist_ok=True)
    with open(os.path.join(scene_dir, "metadata.json"), "w") as f:
        json.dump(scene_data, f, indent=2)

    for obj in objects:
        delete_object(obj)
    return True
