# Minimal YAML reader and ``${...}`` interpolation for config.yaml.
#
# PyYAML is used when available.  BlenderProc runs the generate/render scripts
# inside its own bundled Python, which may not have PyYAML installed, so the
# fallback parser here covers the subset of YAML this file uses.  In practice
# the launchers pass concrete values down as CLI arguments and the BlenderProc
# side rarely reads the config at all.

import os
import re


def _coerce_scalar(text):
    """Convert a YAML scalar to a Python value."""
    text = text.strip()

    if not text:
        return None

    # Strip inline comments outside quotes.
    if text[0] not in "\"'":
        hash_pos = text.find(" #")
        if hash_pos != -1:
            text = text[:hash_pos].strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]

    low = text.lower()
    if low in ("null", "none", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False

    # Inline flow list, e.g. [0, 1, 2]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(p) for p in inner.split(",")]

    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass

    return text


def _parse_yaml_subset(text):
    """Parse the nested mapping / inline-list subset used by config.yaml.

    Deliberately not a general YAML implementation — just enough for this
    file's structure, used only when PyYAML is unavailable.
    """
    root = {}
    # Stack of (indent, container) pairs.
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()

        if ":" not in line:
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            stack = [(-1, root)]

        parent = stack[-1][1]

        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)

    return root


def _load_yaml(path):
    with open(path) as f:
        text = f.read()
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        return _parse_yaml_subset(text)


_VAR_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _interpolate(value, *scopes, _depth=0):
    """Expand ``${name}`` against the given scopes, nearest scope first."""
    if not isinstance(value, str) or "${" not in value:
        return value

    if _depth > 10:
        raise ValueError(f"circular reference in config value: {value!r}")

    def replace(match):
        name = match.group(1)
        for scope in scopes:
            if isinstance(scope, dict) and name in scope:
                resolved = scope[name]
                if resolved is None:
                    return ""
                return str(_interpolate(resolved, *scopes, _depth=_depth + 1))
        # Fall back to the environment, then leave it untouched.
        return os.environ.get(name, match.group(0))

    return _VAR_RE.sub(replace, value)
