# Scene assembly shared by the full and masks-only render paths.

import json
import os

import blenderproc as bproc
import bpy
import numpy as np

from blender.floor import (
    create_bbox_floor_strips,
    remove_bbox_floor_strips,
    setup_floor_from_meta,
)
from blender.objects import (
    align_object_to_floor,
    bbox_dims,
    delete_object,
    load_blend,
    normalize_object,
    set_render_visibility,
)
from blender.renderer import apply_render_settings
from blender.world import setup_bbox_world, setup_lighting_from_meta
from config.profiles import get_profile


def _show_bbox_strips(strips, visible):
    for strip in strips:
        strip.hide_render = not visible


def _show_floor(floor, visible):
    floor.blender_obj.cycles.is_shadow_catcher = visible
    floor.blender_obj.hide_render = not visible


def prepare_bbox_scene(placed_objects, floor, strips):
    """Swap to the bbox pass: objects hidden, checkerboard strips lit flat."""
    set_render_visibility(placed_objects, False)
    _show_floor(floor, False)
    _show_bbox_strips(strips, True)
    setup_bbox_world()


def restore_main_scene(placed_objects, floor, strips, data_root, meta):
    """Undo prepare_bbox_scene and put the HDRI lighting back."""
    _show_bbox_strips(strips, False)
    set_render_visibility(placed_objects, True)
    _show_floor(floor, True)
    setup_lighting_from_meta(data_root, meta)


def _material_surfaces(blender_obj):
    """Principled-BSDF summary of each material slot, for the view JSON."""
    if blender_obj.type != "MESH":
        return []
    surfaces = []
    for sid, mat in enumerate(blender_obj.data.materials):
        if not (mat and mat.use_nodes):
            continue
        bsdf = next(
            (n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None
        )
        if not bsdf:
            continue
        base_color = bsdf.inputs["Base Color"]
        surfaces.append(
            {
                "surface_id": sid,
                "material_name": mat.name,
                "has_texture": base_color.is_linked
                and base_color.links[0].from_node.type == "TEX_IMAGE",
                "material": {
                    "base_color": list(base_color.default_value[:3]),
                    "metallic": float(bsdf.inputs["Metallic"].default_value),
                    "roughness": float(bsdf.inputs["Roughness"].default_value),
                },
            }
        )
    return surfaces


def collect_object_metadata(
    objects, record_bbox_3d=False, record_objaverse_names=False
):
    data = []
    for idx, obj in enumerate(objects):
        bpy.context.view_layer.update()
        blender_obj = obj.blender_obj

        entry = {
            "object_id": idx,
            "name": obj.get_name(),
            "source_file": blender_obj.get("source_file", "unknown"),
            "center": np.mean(np.array(obj.get_bound_box()), axis=0).tolist(),
            "surfaces": _material_surfaces(blender_obj),
        }
        if record_objaverse_names:
            entry["objaverse_uid"] = blender_obj.get("objaverse_uid", "")
            entry["objaverse_name"] = blender_obj.get("objaverse_name", "")
        if record_bbox_3d:
            entry["bbox_3d"] = bbox_dims(obj)

        data.append(entry)
    return data


# Profile keys that decide how an object is loaded and posed.  These MUST
# match the generator that wrote the scene, otherwise the per-view
# rotation_euler in the metadata is interpreted differently and objects come
# out rotated (and their floor contact / bbox shift with them).
GEOMETRY_KEYS = (
    "force_xyz_rotation_mode",
    "apply_rotation_on_normalize",
    "normalize_target_size",
)

_split_warned = {}


def scene_profile(meta, cli_profile):
    """Return the profile a scene must be rendered with.

    The scene's own metadata decides, not --split, so that train scenes are
    always rendered the v6 way and test scenes the v8 way:

      1. "object_loading" block  (written by this generator)
      2. "split" field           (this generator, older revision)
      3. "mode" field present    -> v8 metadata  -> test
      4. none of the above       -> v6 metadata  -> train
    """
    if meta.get("split") in ("train", "test"):
        split = meta["split"]
    elif "mode" in meta:
        split = "test"
    else:
        split = "train"

    profile = get_profile(split)
    loading = meta.get("object_loading") or {}
    for key in GEOMETRY_KEYS:
        if key in loading:
            profile[key] = loading[key]
    if "normalize_target_size" in meta:
        profile["normalize_target_size"] = meta["normalize_target_size"]

    if split != cli_profile["name"] and not _split_warned.get(split):
        _split_warned[split] = True
        print(
            f"  [INFO] metadata says split='{split}' but --split="
            f"'{cli_profile['name']}'; rendering with the '{split}' "
            f"profile so object rotation matches generation"
        )
    return profile


def setup_scene(scene_dir, data_root, quality, profile):
    """Load metadata, build the floor/lighting and load every object.

    Returns (meta, floor, bbox_strips, placed_objects, valid_objects, K,
    profile) or None if the scene has no metadata or no loadable object.
    The returned profile is the one the scene was GENERATED with (see
    scene_profile) and must be used for everything that follows.
    """
    json_path = os.path.join(scene_dir, "metadata.json")
    if not os.path.isfile(json_path):
        return None

    with open(json_path) as f:
        meta = json.load(f)

    bproc.clean_up()
    apply_render_settings(quality)
    setup_lighting_from_meta(data_root, meta)

    floor = bproc.object.create_primitive("PLANE", scale=[30, 30, 1])
    setup_floor_from_meta(data_root, meta, floor)
    floor.blender_obj.cycles.is_shadow_catcher = True

    bbox_strips = create_bbox_floor_strips(floor_stripes=meta.get("floor_stripes"))

    # The normalisation target size is recorded in the metadata; fall back to
    # the profile for metadata written before that field existed.
    profile = scene_profile(meta, profile)
    target_size = profile["normalize_target_size"]
    force_xyz = profile["force_xyz_rotation_mode"]

    placed_objects = []
    for obj_meta in meta["objects"]:
        obj = load_blend(os.path.join(data_root, obj_meta["model_path"]), force_xyz)
        if obj is None:
            print(f"    [WARN] Failed to load {obj_meta['model_path']}")
            placed_objects.append(None)
            continue

        obj.blender_obj["source_file"] = obj_meta["source_file"]
        if profile["record_objaverse_names"]:
            obj.blender_obj["objaverse_uid"] = obj_meta.get("objaverse_uid", "")
            obj.blender_obj["objaverse_name"] = obj_meta.get("objaverse_name", "")
        normalize_object(
            obj, target_size, force_xyz, profile["apply_rotation_on_normalize"]
        )
        placed_objects.append(obj)

    valid_objects = [o for o in placed_objects if o is not None]
    if not valid_objects:
        remove_bbox_floor_strips(bbox_strips)
        return None

    K = bproc.camera.get_intrinsics_as_K_matrix()
    return meta, floor, bbox_strips, placed_objects, valid_objects, K, profile


def pose_objects_for_view(placed_objects, view, force_xyz):
    """Apply the per-view object state from the metadata."""
    for oi, obj in enumerate(placed_objects):
        if obj is None:
            continue
        if obj.blender_obj.animation_data:
            obj.blender_obj.animation_data_clear()
        state = view["objects_state"][oi]
        if force_xyz:
            obj.blender_obj.rotation_mode = "XYZ"
        obj.set_location(state["location"])
        obj.blender_obj.rotation_euler = state["rotation_euler"]
        obj.set_scale(state["scale"])
        align_object_to_floor(obj)
    bpy.context.view_layer.update()


def teardown_scene(placed_objects, bbox_strips):
    for obj in placed_objects:
        if obj is not None:
            delete_object(obj)
    remove_bbox_floor_strips(bbox_strips)
