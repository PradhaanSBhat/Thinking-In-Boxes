# Path helpers shared across the entry points.

import os
import sys


def abspath(value):
    """~-expanded absolute path, or the value unchanged if it is empty."""
    if not isinstance(value, str) or not value:
        return value
    return os.path.abspath(os.path.expanduser(value))


def fail(message):
    """Report a usage error the way the BlenderProc-run scripts do, and stop."""
    print(f"[ERROR] {message}")
    sys.exit(1)
