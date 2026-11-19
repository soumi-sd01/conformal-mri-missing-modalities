import numpy as np


def extract_patch(image, mask, patch_size=(64, 64, 64), strategy="tumor", rng=None, brain_mask=None):
    """Extract aligned 3D MRI and mask patches with strict shape/bounds checks."""
    if rng is None:
        rng = np.random.default_rng()
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError("image must have shape (4, D, H, W)")
    if mask.ndim != 3 or image.shape[1:] != mask.shape:
        raise ValueError("image and mask spatial dimensions must match")
    if strategy not in ("tumor", "brain", "random"):
        raise ValueError("strategy must be 'tumor', 'brain', or 'random'")
    if len(patch_size) != 3 or any(int(p) <= 0 for p in patch_size):
        raise ValueError("patch_size must contain three positive dimensions")
    if any(p > s for p, s in zip(patch_size, mask.shape)):
        raise ValueError("Patch cannot be larger than the input volume")

    if strategy == "tumor":
        candidates = np.argwhere(mask > 0)
        if len(candidates) == 0:
            raise ValueError("Cannot extract a tumor patch: mask has no tumor voxels")
    elif strategy == "brain":
        if brain_mask is None or brain_mask.shape != mask.shape:
            raise ValueError("A correctly shaped brain_mask is required")
        candidates = np.argwhere(brain_mask > 0)
        if len(candidates) == 0:
            raise ValueError("Cannot extract a brain patch: brain mask is empty")
    else:
        candidates = None

    if candidates is None:
        starts = [int(rng.integers(0, s - p + 1)) for p, s in zip(patch_size, mask.shape)]
    else:
        center = candidates[rng.integers(len(candidates))]
        starts = [
            max(0, min(int(c) - p // 2, s - p))
            for c, p, s in zip(center, patch_size, mask.shape)
        ]
    slices = tuple(slice(st, st + sz) for st, sz in zip(starts, patch_size))
    return image[(slice(None),) + slices].copy(), mask[slices].copy()


def extract_training_patch(
    image, mask, brain_mask, patch_size=(64, 64, 64), rng=None, tumor_fraction=0.5
):
    if rng is None:
        rng = np.random.default_rng()
    strategy = "tumor" if rng.random() < tumor_fraction else "brain"
    if strategy == "tumor" and not np.any(mask > 0):
        strategy = "brain"
    return extract_patch(image, mask, patch_size, strategy, rng, brain_mask)
