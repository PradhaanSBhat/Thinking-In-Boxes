# Loading, normalising and placing individual mesh objects.

import bpy
import numpy as np


def load_blend(abs_path, force_xyz=False, unhide=True, warn=True):
    """Append the first mesh object of *abs_path* into the current scene.

    unhide : clear hide_render / hide_viewport on the appended object
             (the render phase does, the generate phase does not).
    warn   : print the exception on failure instead of failing silently.
    """
    try:
        existing = set(bpy.data.objects.keys())
        with bpy.data.libraries.load(abs_path, link=False) as (df, dt):
            dt.objects = df.objects
            dt.meshes = df.meshes
            dt.materials = df.materials

        new_objs = [
            bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in existing
        ]
        if not new_objs:
            return None

        mesh_obj = next((o for o in new_objs if o.type == "MESH"), None)
        if mesh_obj is None:
            for o in new_objs:
                bpy.data.objects.remove(o, do_unlink=True)
            return None

        if mesh_obj.name not in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.link(mesh_obj)
        for o in new_objs:
            if o != mesh_obj:
                try:
                    bpy.data.objects.remove(o, do_unlink=True)
                except Exception:
                    pass

        if unhide:
            mesh_obj.hide_render = False
            mesh_obj.hide_viewport = False
        if force_xyz:
            # Many .blend files use QUATERNION rotation — force XYZ so that
            # rotation_euler assignments from the metadata take effect.
            mesh_obj.rotation_mode = "XYZ"
        mesh_obj.location = (0, 0, 0)
        mesh_obj.rotation_euler = (0, 0, 0)
        mesh_obj.scale = (1, 1, 1)
        bpy.context.view_layer.update()

        from blenderproc.python.types.MeshObjectUtility import MeshObject

        return MeshObject(mesh_obj)
    except Exception as e:
        if warn:
            print(f"  [WARN] Load failed: {e}")
        return None


def delete_object(obj):
    try:
        bo = obj.blender_obj
        md = bo.data
        for col in list(bo.users_collection):
            col.objects.unlink(bo)
        bpy.data.objects.remove(bo, do_unlink=True)
        if md and md.users == 0:
            bpy.data.meshes.remove(md)
    except Exception:
        pass


def normalize_object(obj, target_size, force_xyz=False, apply_rotation=False):
    bo = obj.blender_obj
    if force_xyz:
        bo.rotation_mode = "XYZ"
    # apply_rotation=True bakes rotation+scale into the mesh so geometry
    # sits at identity.  Train keeps rotation un-baked (see config).
    bpy.ops.object.select_all(action="DESELECT")
    bo.select_set(True)
    bpy.context.view_layer.objects.active = bo
    try:
        bpy.ops.object.transform_apply(
            location=False, rotation=apply_rotation, scale=True
        )
    except Exception:
        pass
    bpy.context.view_layer.update()
    pts = np.array(obj.get_bound_box())
    max_dim = max(pts.max(axis=0) - pts.min(axis=0))
    if max_dim < 1e-6:
        return 1.0
    sf = target_size / max_dim
    obj.set_scale([sf] * 3)
    bpy.context.view_layer.update()
    return sf


def align_object_to_floor(obj):
    bpy.context.view_layer.update()
    min_z = min(p[2] for p in obj.get_bound_box())
    loc = obj.get_location()
    obj.set_location([loc[0], loc[1], loc[2] - min_z])


def set_render_visibility(objects, visible):
    """Show or hide *objects* (and their children) in renders."""
    for obj in objects:
        if obj is None:
            continue
        blender_obj = getattr(obj, "blender_obj", obj)
        blender_obj.hide_render = not visible
        for child in blender_obj.children:
            child.hide_render = not visible


def bbox_dims(obj):
    """Axis-aligned 3D bounding box of *obj* in world space."""
    bpy.context.view_layer.update()
    pts = np.array(obj.get_bound_box())
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    dims = maxs - mins
    return {
        "width": float(dims[0]),
        "depth": float(dims[1]),
        "height": float(dims[2]),
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
    }
