# Background image and JSON writing, so rendering never waits on disk.

import json
from concurrent.futures import ThreadPoolExecutor

import imageio


class AsyncWriter:
    def __init__(self, max_workers=4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []

    def write_image(self, path, data):
        self._futures.append(self._executor.submit(imageio.imwrite, path, data))

    def write_json(self, path, obj):
        def _w():
            with open(path, "w") as f:
                json.dump(obj, f, separators=(",", ":"))

        self._futures.append(self._executor.submit(_w))

    def flush(self):
        for fut in self._futures:
            fut.result()
        self._futures.clear()

    def shutdown(self):
        self.flush()
        self._executor.shutdown(wait=True)
