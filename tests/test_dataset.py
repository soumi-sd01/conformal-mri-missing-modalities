import numpy as np
import nibabel as nib
import pytest
import torch
from src.data.dataset import BraTSDataset


@pytest.fixture
def sample_dataset(tmp_path):
    data_dir = tmp_path / "data"
    pid = "BraTS2021_TEST001"
    folder = data_dir / pid
    folder.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for modality in BraTSDataset.MODALITIES:
        volume = rng.normal(10, 2, size=(8, 8, 8)).astype(np.float32)
        volume[:2] = 0
        nib.save(nib.Nifti1Image(volume, np.eye(4)), folder / f"{pid}_{modality}.nii.gz")
    seg = np.zeros((8, 8, 8), dtype=np.uint8)
    seg[2:4] = 1
    seg[4:6] = 2
    seg[6:8] = 4
    nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / f"{pid}_seg.nii.gz")
    split = tmp_path / "train.txt"
    split.write_text(pid + "\n")
    return data_dir, split, folder


def test_shapes_types_and_labels(sample_dataset):
    data_dir, split, _ = sample_dataset
    item = BraTSDataset(data_dir, split)[0]
    assert item["image"].shape == (4, 8, 8, 8)
    assert item["mask"].shape == (8, 8, 8)
    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.int64
    assert set(torch.unique(item["mask"]).tolist()) == {0, 1, 2, 3}


def test_normalization_preserves_background(sample_dataset):
    data_dir, split, _ = sample_dataset
    item = BraTSDataset(data_dir, split)[0]
    for channel in item["image"]:
        assert torch.isfinite(channel).all()
        assert torch.all(channel[:2] == 0)
        vals = channel[channel != 0]
        assert abs(vals.mean().item()) < 1e-5
        assert abs(vals.std(unbiased=False).item() - 1.0) < 1e-5


def test_constant_intensity_becomes_zero(sample_dataset):
    data_dir, split, folder = sample_dataset
    pid = "BraTS2021_TEST001"
    volume = np.full((8, 8, 8), 7.0, dtype=np.float32)
    volume[:2] = 0
    nib.save(nib.Nifti1Image(volume, np.eye(4)), folder / f"{pid}_flair.nii.gz")
    item = BraTSDataset(data_dir, split)[0]
    assert torch.count_nonzero(item["image"][0]) == 0


def test_missing_modality_is_zero_filled(sample_dataset):
    data_dir, split, _ = sample_dataset
    item = BraTSDataset(data_dir, split, missing_modalities=["t1ce"])[0]
    assert not item["modality_mask"][2]
    assert torch.count_nonzero(item["image"][2]) == 0


def test_unknown_segmentation_label_rejected(sample_dataset):
    data_dir, split, folder = sample_dataset
    seg = np.zeros((8, 8, 8), dtype=np.uint8)
    seg[0, 0, 0] = 5
    nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / "BraTS2021_TEST001_seg.nii.gz")
    with pytest.raises(ValueError, match="Unexpected segmentation"):
        BraTSDataset(data_dir, split)[0]
