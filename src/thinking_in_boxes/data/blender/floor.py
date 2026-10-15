# Floor surfaces: the textured shadow-catcher plane and the bbox strips.

import os
import random

import bpy
import numpy as np

# Base checker colours, scaled per strip by its shade value.
_CHECKER_LIGHT = np.array([0.55, 0.52, 0.48])
_CHECKER_DARK = np.array([0.30, 0.28, 0.26])
_REFERENCE_SHADE = 0.3


def _principled_material(name, base_color=None, roughness=0.8):
    """A fresh node-based material with a single Principled BSDF output."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    if base_color is not None:
        bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat, nodes, links, bsdf


def setup_floor_from_meta(data_root, meta, floor):
    if meta.get("floor_texture"):
        blend_path = os.path.join(data_root, meta["floor_texture"])
        existing = set(bpy.data.materials.keys())
        try:
            with bpy.data.libraries.load(blend_path, link=False) as (df, dt):
                dt.materials = df.materials
            new_mats = [m for m in bpy.data.materials if m.name not in existing]
            if new_mats:
                floor.blender_obj.data.materials.clear()
                floor.blender_obj.data.materials.append(new_mats[0])
                return
        except Exception:
            pass

    mat, _, _, _ = _principled_material(
        "FallbackFloor", base_color=(0.5, 0.5, 0.5, 1), roughness=0.8
    )
    floor.blender_obj.data.materials.clear()
    floor.blender_obj.data.materials.append(mat)


def _make_strip(idx, corners, location, shade, checker_scale):
    """One checkerboard floor strip: quad mesh + checker material, hidden."""
    mesh = bpy.data.meshes.new(f"FS_{idx}")
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new(f"FS_{idx}", mesh)
    obj.location = location
    bpy.context.collection.objects.link(obj)

    scale = shade / _REFERENCE_SHADE
    mat, nodes, links, bsdf = _principled_material(f"SCM_{idx}", roughness=0.9)
    checker = nodes.new(type="ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = tuple(
        np.clip(_CHECKER_LIGHT * scale, 0, 1)
    ) + (1.0,)
    checker.inputs["Color2"].default_value = tuple(
        np.clip(_CHECKER_DARK * scale, 0, 1)
    ) + (1.0,)
    checker.inputs["Scale"].default_value = checker_scale
    links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])

    mesh.materials.append(mat)
    obj.hide_render = True
    return obj


def _metadata_strip_corners(floor_stripes, idx, floor_extent):
    """Corners of stripe *idx*, oriented along the recorded stripe direction."""
    direction = np.array(floor_stripes["stripe_dir"])
    perp = np.array([-direction[1], direction[0]])
    near = floor_stripes["boundaries"][idx]
    far = floor_stripes["boundaries"][idx + 1]

    centre = direction * ((near + far) / 2.0)
    half_depth = direction * ((far - near) / 2.0)
    half_width = perp * floor_extent

    return [
        (*(centre - half_depth - half_width), 0),
        (*(centre + half_depth - half_width), 0),
        (*(centre + half_depth + half_width), 0),
        (*(centre - half_depth + half_width), 0),
    ]


def create_bbox_floor_strips(
    floor_stripes=None, floor_extent=30.0, greys=(0.15, 0.3, 0.45)
):
    """Build the checkerboard strips the bbox pass renders instead of the floor.

    With *floor_stripes* from the scene metadata the strips follow the recorded
    direction and shading; without it they fall back to Y-axis aligned bands.
    """
    checker_scale = random.uniform(5.0, 12.0)

    if floor_stripes is not None:
        return [
            _make_strip(
                idx,
                _metadata_strip_corners(floor_stripes, idx, floor_extent),
                (0, 0, 0),
                floor_stripes["shade_values"][idx],
                checker_scale,
            )
            for idx in range(floor_stripes["num_stripes"])
        ]

    # Fallback: original Y-axis aligned strips
    strip_width = (2 * floor_extent) / 3.0
    half_x, half_y = floor_extent, strip_width / 2.0
    corners = [
        (-half_x, -half_y, 0),
        (half_x, -half_y, 0),
        (half_x, half_y, 0),
        (-half_x, half_y, 0),
    ]
    return [
        _make_strip(
            idx,
            corners,
            (0, -floor_extent + strip_width * (idx + 0.5), 0),
            grey,
            checker_scale,
        )
        for idx, grey in enumerate(greys)
    ]


def remove_bbox_floor_strips(strips):
    for obj in strips:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
