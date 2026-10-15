# Collision-aware object placement and camera pose search.

import random

import blenderproc as bproc
import bpy
import numpy as np
from mathutils import Matrix, Vector

from blender.objects import align_object_to_floor

MIN_GAP = 0.2  # metres of clearance required between two objects' XY discs


def get_xy_bounds(obj):
    """The object's XY centre and the radius of the disc that covers it.

    The arithmetic is spelled out per column rather than vectorised over
    axis 0: numpy sums the two forms in a different order, and the last-bit
    difference that produces feeds the collision test, so writing it this way
    keeps regenerated scenes bit-identical.
    """
    pts = np.array(obj.get_bound_box())
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    radius = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2).max()
    return np.array([cx, cy]), radius


def _collides(centre, radius, others, min_gap=MIN_GAP):
    return any(
        np.linalg.norm(centre - other_c) < radius + other_r + min_gap
        for other_c, other_r in others
    )


def try_place_object(obj, placed, min_gap=MIN_GAP, max_attempts=50, spread=2.5):
    """Find a free spot for *obj*; return its (centre, radius) or None.

    The first object lands anywhere in the spread; later ones are pushed out
    from a randomly chosen neighbour so the scene stays clustered.
    """
    for _ in range(max_attempts):
        if not placed:
            position = np.random.uniform(-spread, spread, 2)
        else:
            anchor_c, anchor_r = random.choice(placed)
            obj.set_location([0, 0, 0])
            align_object_to_floor(obj)
            _, own_r = get_xy_bounds(obj)
            angle = random.uniform(0, 2 * np.pi)
            distance = anchor_r + own_r + min_gap + random.uniform(0, 0.3)
            position = anchor_c + distance * np.array([np.cos(angle), np.sin(angle)])

        obj.set_location([position[0], position[1], 0])
        align_object_to_floor(obj)
        centre, radius = get_xy_bounds(obj)
        if not _collides(centre, radius, placed, min_gap):
            return centre, radius
    return None


def is_object_fully_visible(obj, cam_pose, w, h, margin=10):
    """True when all eight bbox corners project inside the image, with margin."""
    bpy.context.scene.camera.matrix_world = Matrix(cam_pose)
    blender_obj = obj.blender_obj
    corners = [blender_obj.matrix_world @ Vector(v) for v in blender_obj.bound_box]
    projected = bproc.camera.project_points(np.array(corners))
    return all(
        margin <= x < w - margin and margin <= y < h - margin for x, y in projected
    )


def compute_camera_poses(all_coords, objects, num_views=3, padding=1.35, max_tries=80):
    """Random orbiting camera poses that keep every object fully in frame."""
    coords = np.array(all_coords)
    centre = coords.mean(axis=0)
    radius = np.linalg.norm(coords - centre, axis=1).max()

    render = bpy.context.scene.render
    w, h = render.resolution_x, render.resolution_y
    fov = bpy.context.scene.camera.data.angle
    # The narrower of the two FOVs bounds how far back the camera must sit.
    min_fov = min(fov, 2 * np.arctan(np.tan(fov / 2) / (w / h)))
    distance = (radius * padding) / np.tan(min_fov / 2)

    base_z = centre[2] + radius * 0.8
    orbit = np.sqrt(max(distance**2 - (base_z - centre[2]) ** 2, radius**2))

    poses = []
    for _ in range(max_tries):
        if len(poses) >= num_views:
            break
        # Draw order matters: it is the random stream that fixes the dataset.
        angle = np.random.uniform(0, 2 * np.pi)
        height = base_z + np.random.uniform(-0.8 * radius, 1.2 * radius)
        this_orbit = orbit + np.random.uniform(-0.3, 0.3) * radius
        location = np.array(
            [
                centre[0] + this_orbit * np.cos(angle),
                centre[1] + this_orbit * np.sin(angle),
                height,
            ]
        )
        target = centre + np.array(
            [
                *np.random.uniform(-0.6 * radius, 0.6 * radius, 2),
                np.random.uniform(-0.2 * radius, 0.4 * radius),
            ]
        )
        rotation = bproc.camera.rotation_from_forward_vec(target - location)
        pose = bproc.math.build_transformation_mat(location.tolist(), rotation)
        if all(is_object_fully_visible(obj, pose, w, h) for obj in objects):
            poses.append(pose)
    return poses


def translate_object_strict(
    obj, objects, placed, cam_pose, w, h, scene_radius, min_gap=MIN_GAP, max_tries=50
):
    """Nudge *obj* to a new spot that collides with nothing and stays in frame.

    Returns True on success; on failure the object is put back where it was.
    *placed* is updated in place for the object that moved.
    """
    original = obj.get_location().copy()
    index = objects.index(obj)
    others = [info for i, info in enumerate(placed) if i != index]
    reach = scene_radius * 0.8

    for _ in range(max_tries):
        offset = [np.random.uniform(-reach, reach), np.random.uniform(-reach, reach), 0]
        obj.set_location((np.array(original) + np.array(offset)).tolist())
        align_object_to_floor(obj)
        centre, radius = get_xy_bounds(obj)
        if _collides(centre, radius, others, min_gap):
            continue
        if not all(is_object_fully_visible(o, cam_pose, w, h) for o in objects):
            continue
        placed[index] = (centre, radius)
        return True

    obj.set_location(original)
    align_object_to_floor(obj)
    return False
