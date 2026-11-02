"""Image geometry uses (height, width); particle coordinates use (y, x)."""
import math


def spatial_size(size):
    if isinstance(size, int):
        return size, size
    height, width = map(int, size)
    if min(height, width) <= 0:
        raise ValueError("Image dimensions must be positive")
    return height, width


def patch_count(size, patch):
    height, width = spatial_size(size)
    if height % patch or width % patch:
        raise ValueError("Image dimensions must be divisible by the prior patch size")
    return (height // patch) * (width // patch)
