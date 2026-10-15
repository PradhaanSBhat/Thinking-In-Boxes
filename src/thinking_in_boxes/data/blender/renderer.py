# Cycles / GPU configuration and the specialised render passes.

from contextlib import contextmanager

import blenderproc as bproc
import bpy

from config.quality import CAMERA_LENS_MM

_depth_enabled = False

# Cycles settings taken from the active quality preset: {preset key: cycles attr}
_QUALITY_TO_CYCLES = {
    "samples": "samples",
    "adaptive_threshold": "adaptive_threshold",
    "adaptive_min_samples": "adaptive_min_samples",
    "max_bounces": "max_bounces",
    "diffuse_bounces": "diffuse_bounces",
    "glossy_bounces": "glossy_bounces",
    "transmission_bounces": "transmission_bounces",
    "clamp_indirect": "sample_clamp_indirect",
}

# Cycles settings that never vary with quality.
_CYCLES_CONSTANTS = {
    "use_denoising": True,
    "denoiser": "OPTIX",
    "use_adaptive_sampling": True,
    "transparent_max_bounces": 2,
    "volume_bounces": 0,
    "caustics_reflective": False,
    "caustics_refractive": False,
    "tile_size": 512,
}


def init_renderer(quality):
    """One-time GPU + renderer setup.  Called once at startup."""
    global _depth_enabled

    bproc.camera.set_resolution(quality["resolution"], quality["resolution"])

    bpy.context.scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()  # only here, never again
    for device in prefs.devices:
        device.use = True

    bproc.camera.set_intrinsics_from_blender_params(
        lens=CAMERA_LENS_MM, lens_unit="MILLIMETERS"
    )
    if not _depth_enabled:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        _depth_enabled = True

    apply_render_settings(quality)


def apply_render_settings(quality):
    """Per-scene render settings.  NO refresh_devices – that's one-time only."""
    scene = bpy.context.scene
    bproc.camera.set_resolution(quality["resolution"], quality["resolution"])
    scene.cycles.device = "GPU"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"

    for key, attr in _QUALITY_TO_CYCLES.items():
        setattr(scene.cycles, attr, quality[key])
    for attr, value in _CYCLES_CONSTANTS.items():
        setattr(scene.cycles, attr, value)

    scene.render.use_persistent_data = True
    bproc.camera.set_intrinsics_from_blender_params(
        lens=CAMERA_LENS_MM, lens_unit="MILLIMETERS"
    )


@contextmanager
def temporary_render_settings(**overrides):
    """Apply Cycles/render overrides for one render, then restore them.

    Keys are looked up on ``scene.cycles`` first, then ``scene.render``, so
    ``samples=1, use_denoising=False, film_transparent=True`` all work.
    """
    scene = bpy.context.scene
    saved = []
    for attr, value in overrides.items():
        holder = scene.cycles if hasattr(scene.cycles, attr) else scene.render
        saved.append((holder, attr, getattr(holder, attr)))
        setattr(holder, attr, value)
    try:
        yield scene
    finally:
        for holder, attr, value in saved:
            setattr(holder, attr, value)


def render_bbox_pass(bbox_samples):
    """Render the checkerboard floor for the bbox overlay, denoising off."""
    with temporary_render_settings(samples=bbox_samples, use_denoising=False):
        return bproc.renderer.render()


def render_depth_pass():
    """Render the full scene for its depth pass only.

    Used by the masks-only path, where the RGB image already exists on disk
    but the depth map needed for occlusion testing does not.  Samples are
    dropped to 1 and denoising disabled: only ``depth`` is read back, so
    shading quality does not matter.
    """
    with temporary_render_settings(samples=1, use_denoising=False):
        return bproc.renderer.render()["depth"][0]


def purge_orphan_data():
    bpy.ops.outliner.orphans_purge(
        do_local_ids=True, do_linked_ids=True, do_recursive=True
    )
    for collection in (bpy.data.images, bpy.data.materials, bpy.data.meshes):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)
