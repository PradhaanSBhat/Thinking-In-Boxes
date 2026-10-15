# Split profiles (train / test).
#
# The train and test datasets were produced by two separate code lines that
# diverged in a number of load-bearing ways.  This module is the single place
# where those differences live, so that the generation and rendering code can
# stay shared.
#
# Do not "unify" the two profiles: the values below reproduce the datasets as
# they were originally generated.  In particular:
#
#   * TRAIN normalises objects to 3.0 units and does NOT bake rotation into the
#     mesh on normalise (transform_apply(rotation=False)).
#   * TEST normalises to 2.5 units, forces rotation_mode='XYZ' and DOES bake
#     rotation (transform_apply(rotation=True)).  The XYZ fix is required
#     because many Objaverse .blend files ship with quaternion rotation, in
#     which case assignments to rotation_euler are silently ignored.
#
# Changing either profile changes the dataset.


# ==========================================================
# TRAIN  (formerly distributed_generation_v6)
# ==========================================================

TRAIN_PROFILE = {
    "name": "train",
    # --- scene composition ---
    "min_obj": 4,
    "max_obj": 4,
    "num_views": 2,
    "normalize_target_size": 3.0,
    # --- per-view perturbation ---
    # Train perturbs a random object per view: always a fresh z-rotation,
    # plus optional scaling / translation of a randomly chosen object.
    "enable_translation": True,
    "enable_scaling": True,
    "enable_rotation": True,
    # --- movement mode ---
    # Train has no mode switch; the camera moves between views AND objects
    # are perturbed.  "mixed" reproduces the original v6 behaviour.
    "mode": "mixed",
    # --- object loading ---
    # v6 did not force XYZ rotation mode and did not bake rotation on
    # normalise.  Kept as-is to preserve train geometry.
    "force_xyz_rotation_mode": False,
    "apply_rotation_on_normalize": False,
    # --- metadata richness ---
    "record_objaverse_names": False,
    "record_bbox_3d": False,
    "record_object_masks": False,
}


# ==========================================================
# TEST  (formerly distributed_generate_v8)
# ==========================================================


TEST_PROFILE = {
    "name": "test",
    # --- scene composition ---
    "min_obj": 1,
    "max_obj": 1,
    "num_views": 2,
    "normalize_target_size": 2.5,
    # --- per-view perturbation ---
    # Test isolates a single factor of variation: only z-rotation is enabled
    # by default.  Translation / scaling are toggled by the ablation scripts.
    "enable_translation": False,
    "enable_scaling": False,
    "enable_rotation": True,
    # --- movement mode ---
    #   camera_movement : camera moves between views, objects stay fixed
    #   object_movement : objects move between views, camera stays fixed
    "mode": "camera_movement",
    # --- object loading ---
    # Required for the rotation fix, see module docstring.
    "force_xyz_rotation_mode": True,
    "apply_rotation_on_normalize": True,
    # --- metadata richness ---
    "record_objaverse_names": True,
    "record_bbox_3d": True,
    "record_object_masks": True,
}


PROFILES = {
    "train": TRAIN_PROFILE,
    "test": TEST_PROFILE,
}


VALID_MODES = ("mixed", "camera_movement", "object_movement")


def get_profile(split, overrides=None):
    """Return a copy of the profile for *split*, with optional overrides.

    Overrides are used by the ablation scripts to flip individual factors
    (e.g. enable only translation) without editing this file.
    """
    if split not in PROFILES:
        raise ValueError(
            f"unknown split '{split}' (expected one of {sorted(PROFILES)})"
        )

    profile = dict(PROFILES[split])

    if overrides:
        unknown = set(overrides) - set(profile)
        if unknown:
            raise ValueError(f"unknown profile keys: {sorted(unknown)}")
        profile.update(overrides)

    if profile["mode"] not in VALID_MODES:
        raise ValueError(
            f"invalid mode '{profile['mode']}' (expected one of {VALID_MODES})"
        )

    return profile


VALID_GENERATOR_FLAGS = (
    "--enable-translation",
    "--no-translation",
    "--enable-scaling",
    "--no-scaling",
    "--enable-rotation",
    "--no-rotation",
)


def normalize_generator_flags(flags):
    """Validate --generator-flag pass-through values.

    generate_scenes.py parses argv leniently and silently ignores anything it
    does not recognise, so a typo would otherwise produce a whole dataset with
    the wrong factors enabled.  Fail loudly here instead.
    """
    out = []
    for flag in flags or []:
        flag = flag.lstrip("=").strip()
        if not flag:
            continue
        if flag not in VALID_GENERATOR_FLAGS:
            raise ValueError(
                f"unknown generator flag '{flag}'; expected one of "
                f"{', '.join(VALID_GENERATOR_FLAGS)}"
            )
        out.append(flag)
    return out
