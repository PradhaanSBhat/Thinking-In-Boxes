# GPU discovery via nvidia-smi.

import subprocess


def detect_gpus():
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return [int(x) for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def gpu_info():
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        r = {}
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) == 3:
                r[int(p[0])] = (int(p[1]), int(p[2]))
        return r
    except Exception:
        return {}
