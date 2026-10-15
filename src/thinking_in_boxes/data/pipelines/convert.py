# GLB → .blend asset converter (phase 0, run once).
#
# For every .glb under --input-root this:
#   1. loads the GLB,
#   2. un-parents its meshes KEEPING their world transform (glTF files keep
#      the up-axis correction on a root node), and applies transforms,
#   3. joins all meshes into a single object,
#   4. moves the origin to the bounding-box centre at the world origin,
#   5. normalises the largest dimension to 2.5,
#   6. packs textures, and
#   7. saves a .blend mirroring the input directory structure.
#
# The scene generator re-normalises each object and drops it onto Z=0 itself,
# so the stored assets only need the correct orientation.
#
# The scene generator's asset index looks for .blend files under a `converted/`
# directory inside each dataset entry, so --output-root should normally point
# somewhere like ``<data-root>/<DatasetName>/converted/...``.

import os
import sys
from glob import glob

import blenderproc as bproc
import bpy
import numpy as np
from mathutils import Vector
from tqdm import tqdm

from config.argv import flag, parse_argv, value
from config.paths import abspath

OPTIONS = {
    "--input-root": value("input_root"),
    "--output-root": value("output_root"),
    "--glob": value("pattern"),
    "--config": value("config_path"),
    "--overwrite": flag("overwrite"),
    "--prune": flag("prune"),
}

DEFAULT_PATTERN = os.path.join("*", "*.glb")  # <hash-prefix>/<uid>.glb

# Size the stored assets are normalised to (matches the original
# preconvert_worker.py).  Generation re-normalises per split regardless.
CONVERT_TARGET_SIZE = 2.5


def parse_args():
    args = parse_argv(sys.argv, OPTIONS)
    input_root = args.get("input_root")
    output_root = args.get("output_root")
    pattern = args.get("pattern")

    # Fall back to the `assets` section of config.yaml for anything not given.
    if not (input_root and output_root and pattern):
        try:
            from config.settings import load_settings

            assets = load_settings(args.get("config_path")).assets
            input_root = input_root or assets.get("glb_input")
            output_root = output_root or assets.get("blend_output")
            pattern = pattern or assets.get("pattern")
        except Exception as e:
            print(f"  [WARN] could not read config: {e}")

    for name, path, setting in (
        ("--input-root", input_root, "assets.glb_input"),
        ("--output-root", output_root, "assets.blend_output"),
    ):
        if not path:
            print(f"[ERROR] {name} required (or set {setting} in config.yaml)")
            sys.exit(1)

    return (
        abspath(input_root),
        abspath(output_root),
        pattern or DEFAULT_PATTERN,
        args.get("overwrite", False),
        args.get("prune", False),
    )


def process_glb(glb_path, output_path):
    """Convert one model.  Returns True on success, False if it has no mesh.

    This reproduces the conversion that produced the original `converted/`
    assets (render_scripts/preconvert_worker.py).  Two details are
    load-bearing for objects standing upright on the floor:

      * Meshes are un-parented with CLEAR_KEEP_TRANSFORM *before* the
        transforms are applied.  glTF files (Sketchfab ones in particular)
        keep their up-axis correction on a root empty; dropping that parent
        without keeping the transform leaves the object lying on its side.
      * No "smallest dimension is up" heuristic.  The glTF importer already
        converts Y-up to Blender's Z-up, so re-rotating flat objects tips
        them over.
    """
    ext = os.path.splitext(glb_path)[1].lower()

    # Clean slate
    bpy.ops.wm.read_factory_settings(use_empty=True)

    existing = set(bpy.data.objects.keys())

    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=glb_path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=glb_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=glb_path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=glb_path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=glb_path)
    else:
        raise ValueError(f"unsupported extension: {ext}")

    new_objs = [bpy.data.objects[n] for n in bpy.data.objects.keys()
                if n not in existing]
    if not new_objs:
        print("[WARN] No objects imported, skipping ...")
        return False

    # ----------------------------------------
    # 1. Clear animations
    # ----------------------------------------
    for bo in new_objs:
        if bo.animation_data:
            bo.animation_data_clear()

    mesh_objs = [o for o in new_objs if o.type == 'MESH']
    non_mesh  = [o for o in new_objs if o.type != 'MESH']

    if not mesh_objs:
        for o in non_mesh:
            bpy.data.objects.remove(o, do_unlink=True)
        print("[WARN] No meshes found, skipping ...")
        return False

    def select_only(objs, active):
        bpy.ops.object.select_all(action='DESELECT')
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = active

    # ----------------------------------------
    # 2. Unparent, keeping the world transform
    # ----------------------------------------
    select_only(mesh_objs, mesh_objs[0])
    try:
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    except Exception:
        pass

    # ----------------------------------------
    # 3. Apply transforms
    # ----------------------------------------
    select_only(mesh_objs, mesh_objs[0])
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception:
        pass

    # ----------------------------------------
    # 4. Join meshes
    # ----------------------------------------
    if len(mesh_objs) > 1:
        select_only(mesh_objs, mesh_objs[0])
        bpy.ops.object.join()

    joined = mesh_objs[0]
    joined.name = os.path.splitext(os.path.basename(glb_path))[0]

    # ----------------------------------------
    # 5. Origin to bounding-box centre, at world origin
    # ----------------------------------------
    select_only([joined], joined)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    joined.location = (0, 0, 0)
    joined.rotation_euler = (0, 0, 0)
    joined.scale = (1, 1, 1)

    # ----------------------------------------
    # 6. Normalise scale
    # ----------------------------------------
    # The scene generator re-normalises to the split's target size anyway;
    # this just keeps the stored assets at a sane, uniform size.
    pts = np.array([joined.matrix_world @ Vector(c) for c in joined.bound_box])
    max_dim = max(pts.max(axis=0) - pts.min(axis=0))

    if max_dim > 1e-6:
        sf = CONVERT_TARGET_SIZE / max_dim
        joined.scale = (sf, sf, sf)

        select_only([joined], joined)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        joined.location = (0, 0, 0)

    # ----------------------------------------
    # 7. Remove non-mesh leftovers
    # ----------------------------------------
    for o in non_mesh:
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass

    joined["source_file"] = os.path.basename(glb_path)
    joined["original_path"] = glb_path

    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True,
                                   do_recursive=True)

    # ----------------------------------------
    # 8. Pack textures and save (atomic: .tmp then rename)
    # ----------------------------------------
    try:
        bpy.ops.file.pack_all()
    except Exception:
        pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + f".tmp.{os.getpid()}"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=tmp_path, compress=True)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return True


def prune_source(glb_path, input_root):
    """Delete a converted .glb, and its directory once that empties out.

    Only ever called after the matching .blend exists, so the asset the
    pipeline actually consumes is never the one removed.
    """
    try:
        os.remove(glb_path)
    except OSError as e:
        tqdm.write(f"[WARN] could not remove {glb_path}: {e}")
        return
    parent = os.path.dirname(glb_path)
    if os.path.abspath(parent) != input_root and not os.listdir(parent):
        try:
            os.rmdir(parent)
        except OSError:
            pass


def main():
    input_root, output_root, pattern, overwrite, prune = parse_args()

    # INIT ONCE
    bproc.init()

    glb_files = sorted(glob(os.path.join(input_root, pattern)))
    print(f"Found {len(glb_files)} GLB files under {input_root}")
    if not glb_files:
        print("[ERROR] Nothing to convert. Check --input-root and --glob.")
        sys.exit(1)
    if prune:
        print("[PRUNE] each .glb is deleted once its .blend has been written")

    succeeded = failed = skipped = pruned = 0

    for glb_path in tqdm(glb_files, desc="Converting GLB -> Blend", unit="file"):
        relative = os.path.splitext(os.path.relpath(glb_path, input_root))[0] + ".blend"
        output_path = os.path.join(output_root, relative)

        if os.path.isfile(output_path) and not overwrite:
            skipped += 1
            # A resumed run still tidies up sources converted before --prune.
            if prune:
                prune_source(glb_path, input_root)
                pruned += 1
            continue

        try:
            converted = process_glb(glb_path, output_path)
        except Exception as e:
            converted = False
            tqdm.write(f"[FAIL] {glb_path} - {e}")

        if converted:
            succeeded += 1
            if prune:
                prune_source(glb_path, input_root)
                pruned += 1
        else:
            failed += 1

    print(
        f"\nDone. {succeeded} succeeded, {failed} failed, "
        f"{skipped} skipped (already converted) "
        f"out of {len(glb_files)} files."
    )
    if prune:
        print(f"[PRUNE] removed {pruned} source .glb files")
