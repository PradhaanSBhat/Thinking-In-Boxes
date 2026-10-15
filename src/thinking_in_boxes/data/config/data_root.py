# Asset data root resolution (never hard-coded).

import os

from config.paths import abspath

DATA_ROOT_ENV_VAR = "DATA_ROOT"


def resolve_data_root(cli_value=None, required=True):
    """Resolve the asset data root.

    Precedence: explicit --data-root argument, then config.yaml, then the
    DATA_ROOT environment variable.  There is deliberately no built-in
    default, since the correct path is machine-specific.

    The launchers normally pass --data-root explicitly, so this only does
    real work when a phase script is invoked directly via `blenderproc run`.
    Reading the config is best-effort: BlenderProc's bundled Python may lack
    PyYAML, and the environment variable still covers that case.
    """
    value = cli_value or None

    if not value:
        try:
            from config.settings import load_settings

            value = load_settings().data_root
        except Exception:
            value = None

    if not value:
        value = os.environ.get(DATA_ROOT_ENV_VAR) or None

    if value:
        return abspath(value)

    if required:
        raise ValueError(
            "no data root given: pass --data-root /path/to/datasets, set "
            f"`paths.data_root` in config.yaml, or set the "
            f"{DATA_ROOT_ENV_VAR} environment variable"
        )
    return None
