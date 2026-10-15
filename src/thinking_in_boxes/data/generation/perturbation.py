# Per-view object perturbation, one function per movement mode.

import random

import bpy
import numpy as np

from blender.objects import align_object_to_floor
from generation.placement import translate_object_strict

SCALE_RANGE = (0.5, 1.5)


def _random_scale(base_scale):
    return [base_scale * random.uniform(*SCALE_RANGE)] * 3


def _mutate_and_realign(objects, indices, mutate):
    """Apply *mutate* to each index, then re-seat those objects on the floor."""
    for idx in indices:
        mutate(idx)
    bpy.context.view_layer.update()
    for idx in indices:
        align_object_to_floor(objects[idx])


def perturb_mixed(objects, base_scales, bounds, pose, w, h, scene_radius, profile):
    """Train-split perturbation (original v6 behaviour).

    Picks ONE random object and gives it a fresh z-rotation, then optionally
    scales one random object and translates one random object.  Each of the
    three picks is independent.
    """
    rotate_idx = random.randint(0, len(objects) - 1)
    objects[rotate_idx].set_rotation_euler([0, 0, random.uniform(0, 2 * np.pi)])
    align_object_to_floor(objects[rotate_idx])

    if profile["enable_scaling"]:
        scale_idx = random.randint(0, len(objects) - 1)
        objects[scale_idx].set_scale(_random_scale(base_scales[scale_idx]))
        align_object_to_floor(objects[scale_idx])

    if profile["enable_translation"]:
        translate_object_strict(
            random.choice(objects), objects, bounds, pose, w, h, scene_radius
        )


def perturb_object_movement(
    objects, base_scales, bounds, pose, w, h, scene_radius, profile
):
    """Test-split object_movement perturbation (original v8 behaviour).

    Picks a random SUBSET of objects and applies the enabled factors
    (rotation / scaling / translation) to every object in that subset.
    """
    num_to_move = random.randint(1, len(objects))
    move_indices = random.sample(range(len(objects)), num_to_move)

    if profile["enable_rotation"]:
        _mutate_and_realign(
            objects,
            move_indices,
            lambda i: setattr(
                objects[i].blender_obj,
                "rotation_euler",
                (0, 0, random.uniform(0, 2 * np.pi)),
            ),
        )

    if profile["enable_scaling"]:
        _mutate_and_realign(
            objects,
            move_indices,
            lambda i: objects[i].set_scale(_random_scale(base_scales[i])),
        )

    # Translated one at a time, so each move is collision-checked in turn.
    if profile["enable_translation"]:
        for idx in move_indices:
            translate_object_strict(
                objects[idx], objects, bounds, pose, w, h, scene_radius
            )
