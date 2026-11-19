import numpy as np
import pytest
from src.data.patches import extract_patch


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    image = rng.normal(size=(4, 32, 32, 32)).astype(np.float32)
    mask = np.zeros((32, 32, 32), dtype=np.int64)
    mask[12:18, 12:18, 12:18] = 1
    brain = np.ones((32, 32, 32), dtype=bool)
    return image, mask, brain


def test_tumor_patch_shape_and_tumor_presence(sample):
    image, mask, _ = sample
    x, y = extract_patch(image, mask, (16, 16, 16), "tumor", np.random.default_rng(42))
    assert x.shape == (4, 16, 16, 16) and y.shape == (16, 16, 16)
    assert np.any(y > 0)


def test_brain_patch(sample):
    image, mask, brain = sample
    x, y = extract_patch(image, mask, (16, 16, 16), "brain", np.random.default_rng(1), brain)
    assert x.shape == (4, 16, 16, 16) and y.shape == (16, 16, 16)


def test_random_patch(sample):
    image, mask, _ = sample
    x, y = extract_patch(image, mask, (16, 16, 16), "random", np.random.default_rng(1))
    assert x.shape == (4, 16, 16, 16)


def test_invalid_strategy_rejected(sample):
    image, mask, _ = sample
    with pytest.raises(ValueError, match="strategy"):
        extract_patch(image, mask, strategy="bad")


def test_patch_larger_than_volume_rejected(sample):
    image, mask, _ = sample
    with pytest.raises(ValueError, match="larger"):
        extract_patch(image, mask, (64, 64, 64), "random")


def test_tumor_strategy_rejects_empty_mask(sample):
    image, _, _ = sample
    with pytest.raises(ValueError, match="no tumor voxels"):
        extract_patch(image, np.zeros((32, 32, 32)), (16, 16, 16), "tumor")


def test_invalid_image_mask_alignment(sample):
    image, mask, _ = sample
    with pytest.raises(ValueError, match="dimensions"):
        extract_patch(image, mask[:10], (8, 8, 8), "random")
