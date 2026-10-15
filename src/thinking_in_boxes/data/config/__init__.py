# Configuration: split profiles, quality presets and config.yaml.

from config.data_root import DATA_ROOT_ENV_VAR, resolve_data_root
from config.profiles import (
    PROFILES,
    TEST_PROFILE,
    TRAIN_PROFILE,
    VALID_GENERATOR_FLAGS,
    VALID_MODES,
    get_profile,
    normalize_generator_flags,
)
from config.quality import (
    CAMERA_LENS_MM,
    FLUSH_INTERVAL,
    ORPHAN_PURGE_INTERVAL,
    OUTPUT_RESOLUTION,
    QUALITY_FAST,
    QUALITY_NORMAL,
    RENDER_RESOLUTION,
)
from config.settings import CONFIG_FILENAME, Settings, load_settings

__all__ = [
    "CAMERA_LENS_MM",
    "CONFIG_FILENAME",
    "DATA_ROOT_ENV_VAR",
    "FLUSH_INTERVAL",
    "ORPHAN_PURGE_INTERVAL",
    "OUTPUT_RESOLUTION",
    "PROFILES",
    "QUALITY_FAST",
    "QUALITY_NORMAL",
    "RENDER_RESOLUTION",
    "Settings",
    "TEST_PROFILE",
    "TRAIN_PROFILE",
    "VALID_GENERATOR_FLAGS",
    "VALID_MODES",
    "get_profile",
    "load_settings",
    "normalize_generator_flags",
    "resolve_data_root",
]
