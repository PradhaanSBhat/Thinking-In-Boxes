# config.yaml loader.
#
# Single source of truth for every runtime setting.  Resolution order:
#
#     CLI argument  >  config.yaml  >  built-in default
#
# ``config.yaml`` supports ``${name}`` interpolation between entries, so paths
# can be written in terms of one another::
#
#     output_root:  ./output
#     train_scenes: ${output_root}/scenes_train
#
# Names resolve against the enclosing section first, then `paths`, then the
# top level, so ``${data_root}`` works from inside the `assets` section.

import os
import re
import sys

from config.paths import abspath
from config.project import PROJECT_ROOT
from config.yaml_lite import _interpolate, _load_yaml

CONFIG_FILENAME = "config.yaml"

# Where the config lives by default: at the repository root.
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, CONFIG_FILENAME)

# Environment variable allowing a different config file without editing code.
CONFIG_ENV_VAR = "DATASET_CONFIG"

# The placeholder shipped in the example config; treated as "not configured".
_PLACEHOLDER_DATA_ROOTS = {"/path/to/datasets", "path/to/datasets"}


DEFAULTS = {
    "paths": {
        "data_root": None,
        "output_root": "./output",
        "train_scenes": "${output_root}/scenes_train",
        "test_scenes_camera": "${output_root}/scenes_test_camera",
        "test_scenes_object": "${output_root}/scenes_test_object",
        "ablation_root": "${output_root}/ablations",
        "log_root": "${output_root}/logs",
    },
    "scale": {
        "train_start": 0,
        "train_scenes": 20000,
        "test_scenes": 500,
        "num_workers": 40,
    },
    "render": {
        "gpu_ids": [],
        "procs_per_gpu": 5,
        "batch_size": 50,
        "fast": False,
        "masks": None,
    },
    "assets": {
        "dataset": "ObjaverseXL_sketchfab",
        "glb_input": "${data_root}/${dataset}/raw/hf-objaverse-v1/glbs",
        "blend_output": "${data_root}/${dataset}/converted",
        "pattern": "*/*.glb",
    },
    "objects": {
        "train_min": None,
        "train_max": None,
        "test_min": None,
        "test_max": None,
    },
    "ablations": {},
}


_PATH_KEYS = set(DEFAULTS["paths"])


class Settings:
    """Resolved configuration, with attribute access per section."""

    def __init__(self, data, source_path=None):
        self._data = data
        self.source_path = source_path

    # --- section access ---------------------------------------------------
    @property
    def paths(self):
        return self._data["paths"]

    @property
    def scale(self):
        return self._data["scale"]

    @property
    def render(self):
        return self._data["render"]

    @property
    def assets(self):
        return self._data["assets"]

    @property
    def objects(self):
        return self._data["objects"]

    @property
    def ablations(self):
        return self._data["ablations"]

    # --- convenience ------------------------------------------------------
    @property
    def data_root(self):
        return self.paths.get("data_root")

    def require_data_root(self):
        """Return data_root, creating it if missing, or raise if it is unset."""
        value = self.data_root
        if not value:
            raise ValueError(
                "no data_root configured.\n"
                f"  Set `paths.data_root` in {self.source_path or CONFIG_FILENAME}, "
                "or pass --data-root /path/to/datasets."
            )
        if value in _PLACEHOLDER_DATA_ROOTS or "/path/to/" in value:
            raise ValueError(
                f"`paths.data_root` is still the example placeholder "
                f"({value}).\n"
                f"  Edit {self.source_path or CONFIG_FILENAME} and point it at "
                "your assets."
            )
        if not os.path.isdir(value):
            try:
                os.makedirs(value)
            except OSError as e:
                raise ValueError(
                    f"`paths.data_root` does not exist and could not be "
                    f"created:\n  {value}\n  {e}"
                ) from e
            print(f"[SETUP] Created data_root: {value}", file=sys.stderr)
        return value

    def scene_dir_for(self, split, mode=None):
        """Default scene directory for a split / test mode."""
        if split == "train":
            return self.paths["train_scenes"]
        if mode == "object_movement":
            return self.paths["test_scenes_object"]
        return self.paths["test_scenes_camera"]

    def scenes_for(self, split):
        """Default scene count for a split."""
        key = "train_scenes" if split == "train" else "test_scenes"
        return self.scale[key]

    def start_for(self, split):
        """Default first scene index for a split (test always starts at 0)."""
        return self.scale["train_start"] if split == "train" else 0

    def object_range(self, split, cli_min=None, cli_max=None):
        """(min_obj, max_obj) for a split.

        CLI value > config.yaml `objects.<split>_min/_max` > split profile
        default in config/profiles.py.  Raises ValueError on an invalid range.
        """
        from config.profiles import get_profile

        profile = get_profile(split)
        objs = self.objects or {}

        def pick(cli, key, default):
            if cli is not None:
                return cli
            value = objs.get(f"{split}_{key}")
            return default if value is None else value

        lo = int(pick(cli_min, "min", profile["min_obj"]))
        hi = int(pick(cli_max, "max", profile["max_obj"]))
        if lo < 1:
            raise ValueError(f"{split} min objects must be >= 1, got {lo}")
        if hi < lo:
            raise ValueError(f"{split} max objects ({hi}) is smaller than min ({lo})")
        return lo, hi

    def gpu_ids(self):
        """GPU ids as a list of ints; empty means auto-detect."""
        raw = self.render.get("gpu_ids") or []
        if isinstance(raw, str):
            raw = [p for p in re.split(r"[,\s]+", raw) if p]
        elif not isinstance(raw, (list, tuple)):
            raw = [raw]
        return [int(x) for x in raw]

    def __repr__(self):
        return f"<Settings from {self.source_path}>"


def _deep_merge(base, override):
    """Recursively merge *override* into a copy of *base*."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(config_path=None, overrides=None):
    """Load and resolve configuration.

    config_path : explicit path, else $DATASET_CONFIG, else config.yaml
                  beside this module.
    overrides   : {section: {key: value}} applied last (CLI arguments).
                  None values are ignored so unset flags do not clobber
                  configured values.
    """
    path = config_path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH

    loaded = {}
    resolved_path = None
    if path and os.path.isfile(path):
        loaded = _load_yaml(path)
        resolved_path = os.path.abspath(path)
    elif config_path:
        # An explicitly requested config that does not exist is an error.
        raise ValueError(f"config file not found: {config_path}")

    data = _deep_merge(DEFAULTS, loaded)

    # Apply CLI overrides, skipping None (= flag not supplied).
    for section, values in (overrides or {}).items():
        if not values:
            continue
        clean = {k: v for k, v in values.items() if v is not None}
        if clean:
            data[section] = _deep_merge(data.get(section, {}), clean)

    # Environment still works as a convenience for data_root.
    if not data["paths"].get("data_root"):
        env_root = os.environ.get("DATA_ROOT")
        if env_root:
            data["paths"]["data_root"] = env_root

    # Interpolate: paths first (they feed everything else), then the rest.
    paths = data["paths"]
    for key, value in list(paths.items()):
        paths[key] = _interpolate(value, paths, data)
    for key in list(paths):
        if key in _PATH_KEYS:
            paths[key] = abspath(paths[key])

    for section_name in ("assets", "scale", "render"):
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, value in list(section.items()):
            section[key] = _interpolate(value, section, paths, data)

    for key in ("glb_input", "blend_output"):
        if data["assets"].get(key):
            data["assets"][key] = abspath(data["assets"][key])

    return Settings(data, resolved_path)
