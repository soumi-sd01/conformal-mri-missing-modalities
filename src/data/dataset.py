from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """BraTS 2021 loader. Native geometry is preserved; no resampling is performed."""

    MODALITIES = ["flair", "t1", "t1ce", "t2"]
    LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}

    def __init__(self, data_dir, split_file, normalize=True, missing_modalities=None):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.missing_modalities = set(missing_modalities or [])
        invalid = self.missing_modalities - set(self.MODALITIES)
        if invalid:
            raise ValueError(f"Invalid modality names: {sorted(invalid)}")
        if len(self.missing_modalities) == len(self.MODALITIES):
            raise ValueError("At least one modality must remain available.")
        split_path = Path(split_file)
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        self.patient_ids = [x.strip() for x in split_path.read_text().splitlines() if x.strip()]
        if not self.patient_ids:
            raise ValueError(f"Split file is empty: {split_path}")

    @staticmethod
    def normalize_volume(volume):
        """Nonzero-voxel z-score; constant nonzero volumes become all zero."""
        volume = np.asarray(volume, dtype=np.float32).copy()
        mask = volume != 0
        if not mask.any():
            return volume
        values = volume[mask]
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError("Non-finite MRI intensity statistics.")
        if std <= 1e-8:
            volume[mask] = 0.0
        else:
            volume[mask] = (values - mean) / std
        if not np.isfinite(volume).all():
            raise ValueError("MRI volume contains NaN or Inf after normalization.")
        return volume

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        patient_dir = self.data_dir / patient_id
        loaded = {}
        reference_shape = None
        reference_affine = None

        for modality in self.MODALITIES:
            if modality in self.missing_modalities:
                continue
            path = patient_dir / f"{patient_id}_{modality}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing modality file: {path}")
            nii = nib.load(str(path))
            volume = np.asarray(nii.dataobj, dtype=np.float32)
            if reference_shape is None:
                reference_shape = volume.shape
                reference_affine = nii.affine
            else:
                if volume.shape != reference_shape:
                    raise ValueError(
                        f"Shape mismatch for {patient_id}/{modality}: {volume.shape} != {reference_shape}"
                    )
                if not np.allclose(nii.affine, reference_affine, atol=1e-4):
                    raise ValueError(f"Affine mismatch for {patient_id}/{modality}")
            if self.normalize:
                volume = self.normalize_volume(volume)
            elif not np.isfinite(volume).all():
                raise ValueError(f"Non-finite values in {path}")
            loaded[modality] = volume

        if reference_shape is None:
            raise RuntimeError("No MRI modalities were loaded.")
        seg_path = patient_dir / f"{patient_id}_seg.nii.gz"
        if not seg_path.is_file():
            raise FileNotFoundError(f"Missing segmentation: {seg_path}")
        seg_nii = nib.load(str(seg_path))
        seg = np.asarray(seg_nii.dataobj)
        if seg.shape != reference_shape:
            raise ValueError(f"Segmentation shape mismatch for {patient_id}")
        if not np.allclose(seg_nii.affine, reference_affine, atol=1e-4):
            raise ValueError(f"Segmentation affine mismatch for {patient_id}")
        unique = set(np.unique(seg).tolist())
        if not unique.issubset(self.LABEL_MAP):
            raise ValueError(f"Unexpected segmentation labels for {patient_id}: {sorted(unique)}")

        channels, modality_mask = [], []
        for modality in self.MODALITIES:
            available = modality not in self.missing_modalities
            channels.append(loaded[modality] if available else np.zeros(reference_shape, dtype=np.float32))
            modality_mask.append(available)

        mapped = np.zeros(seg.shape, dtype=np.int64)
        for raw_label, mapped_label in self.LABEL_MAP.items():
            mapped[seg == raw_label] = mapped_label

        image = np.stack(channels, axis=0).astype(np.float32, copy=False)
        return {
            "image": torch.from_numpy(image.copy()),
            "mask": torch.from_numpy(mapped.copy()),
            "modality_mask": torch.tensor(modality_mask, dtype=torch.bool),
            "patient_id": patient_id,
        }
