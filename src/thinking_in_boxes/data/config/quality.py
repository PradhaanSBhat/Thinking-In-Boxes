# Render quality presets and the constants shared by both phases.

QUALITY_NORMAL = {
    "resolution": 512,
    "samples": 32,
    "adaptive_threshold": 0.08,
    "adaptive_min_samples": 8,
    "max_bounces": 3,
    "diffuse_bounces": 2,
    "glossy_bounces": 2,
    "transmission_bounces": 1,
    "clamp_indirect": 10.0,
    "bbox_samples": 4,
}


QUALITY_FAST = {
    "resolution": 256,
    "samples": 16,
    "adaptive_threshold": 0.15,
    "adaptive_min_samples": 4,
    "max_bounces": 2,
    "diffuse_bounces": 1,
    "glossy_bounces": 1,
    "transmission_bounces": 0,
    "clamp_indirect": 5.0,
    "bbox_samples": 2,
}


OUTPUT_RESOLUTION = 512


CAMERA_LENS_MM = 65


RENDER_RESOLUTION = 512


ORPHAN_PURGE_INTERVAL = 50  # purge is CPU-expensive; do it rarely


FLUSH_INTERVAL = 5  # flush the async writer every N scenes
