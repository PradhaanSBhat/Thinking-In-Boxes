# World lighting: HDRI environment for the RGB pass, flat grey for bbox.

import os

import blenderproc as bproc
import bpy


def _clear_lights():
    for obj in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj, do_unlink=True)


def _fresh_world_nodes():
    """Empty the world's node tree and return its (nodes, links)."""
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    return nodes, links


def setup_lighting_from_meta(data_root, meta):
    """HDRI environment lighting, rotated and scaled as the metadata records."""
    _clear_lights()
    nodes, links = _fresh_world_nodes()

    environment = nodes.new("ShaderNodeTexEnvironment")
    background = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    mapping = nodes.new("ShaderNodeMapping")
    coordinates = nodes.new("ShaderNodeTexCoord")

    hdri = meta["hdri"]
    environment.image = bpy.data.images.load(os.path.join(data_root, hdri["path"]))
    mapping.inputs["Rotation"].default_value[2] = hdri["rotation_z"]
    background.inputs["Strength"].default_value = hdri["strength"]

    links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    fill = bproc.types.Light()
    fill.set_type("POINT")
    fill.set_location([3, -3, 5])
    fill.set_energy(150)


def setup_bbox_world():
    """Flat grey, unlit world so the bbox pass renders the strips evenly."""
    nodes, links = _fresh_world_nodes()
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    background.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(background.outputs["Background"], output.inputs["Surface"])
    _clear_lights()
