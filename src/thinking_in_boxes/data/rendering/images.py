# Small image conversions shared by the render passes.

import cv2
import numpy as np


def resized(img, output_res, interpolation):
    """Resize to output_res x output_res, or return *img* if already that size."""
    if img.shape[0] == output_res:
        return img
    return cv2.resize(img, (output_res, output_res), interpolation=interpolation)


def to_rgb8(img, output_res):
    """Float render result -> uint8 RGB at the output resolution."""
    return resized(
        np.clip(img, 0, 255).astype(np.uint8), output_res, cv2.INTER_LANCZOS4
    )


def to_depth_mm(depth, output_res):
    """Depth in metres -> uint16 millimetres at the output resolution."""
    millimetres = np.where(np.isfinite(depth), depth * 1000, 0).astype(np.uint16)
    return resized(millimetres, output_res, cv2.INTER_NEAREST)


def to_mask8(mask, output_res):
    """Binary mask at the output resolution, nearest-neighbour."""
    return resized(mask, output_res, cv2.INTER_NEAREST)
