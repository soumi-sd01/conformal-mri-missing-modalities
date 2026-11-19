from concurrent.futures import ProcessPoolExecutor
import gc
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset
from src.data.dataset import BraTSDataset
from src.data.patches import extract_training_patch


def _resolve_patches_per_patient(requested, n_patients, patch_size):
    if isinstance(requested, int) and requested > 0:
        return int(requested)
    avail_gib = psutil.virtual_memory().available / (1024 ** 3)
    # Exact footprint per 64^3 patch: 4*64^3*2B (fp16 image) + 64^3*1B (uint8 mask) = 2.25 MiB
    voxels = int(patch_size[0]) * int(patch_size[1]) * int(patch_size[2])
    bytes_per_patch = voxels * (4 * 2 + 1)
    # Keep comfortable RAM headroom on Colab (~12 GB runtime)
    if avail_gib < 7.0:
        k = 1
    elif avail_gib < 10.5:
        k = 2
    else:
        k = 4
    est_gib = (n_patients * k * bytes_per_patch) / (1024 ** 3)
    print(f"RAM guard: {avail_gib:.2f} GiB free -> patches_per_patient={k} (~{est_gib:.2f} GiB cache)")
    return k


def _cache_worker(args):
    data_dir, split_file, normalize, patch_size, deterministic, seed, tumor_fraction, k, idx = args
    base = BraTSDataset(data_dir, split_file, normalize=normalize)
    sample = base[idx]
    image = sample["image"].numpy()
    mask = sample["mask"].numpy()
    brain_mask = np.any(image != 0, axis=0)
    patches = []
    for p_i in range(k):
        rng = np.random.default_rng(seed + idx * 31 + p_i)
        x_p, y_p = extract_training_patch(
            image, mask, brain_mask, patch_size, rng, tumor_fraction
        )
        patches.append((x_p.astype(np.float16), y_p.astype(np.uint8)))
    return idx, patches, sample["patient_id"]


class BraTSPatchDataset(Dataset):
    """Pre-caches k cropped 64^3 patches per patient in RAM (~1.65 GiB per k=1, ~6.59 GiB at k=4 for 750 patients)."""

    def __init__(
        self,
        data_dir,
        split_file,
        patch_size=(64, 64, 64),
        normalize=True,
        deterministic=False,
        seed=42,
        tumor_fraction=0.5,
        patches_per_patient="auto",
    ):
        self.base = BraTSDataset(data_dir, split_file, normalize=normalize)
        self.patch_size = tuple(patch_size)
        self.deterministic = deterministic
        self.seed = int(seed)
        self.tumor_fraction = float(tumor_fraction)
        n = len(self.base)
        self.k = 1 if deterministic else _resolve_patches_per_patient(patches_per_patient, n, self.patch_size)
        self.cache = [None] * n

        tasks = [
            (
                str(data_dir),
                str(split_file),
                normalize,
                self.patch_size,
                deterministic,
                self.seed,
                self.tumor_fraction,
                self.k,
                idx,
            )
            for idx in range(n)
        ]
        print(f"Caching {n} patients x {self.k} patches ({self.patch_size}) in RAM...")
        with ProcessPoolExecutor(max_workers=2) as ex:
            for done, (idx, patches, pid) in enumerate(ex.map(_cache_worker, tasks, chunksize=10), 1):
                self.cache[idx] = (patches, pid)
                if done % 250 == 0 or done == n:
                    print(f"  Cached {done}/{n} patients")
        gc.collect()
        print("Patch cache ready in RAM.")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        patches, pid = self.cache[index]
        pick = 0 if self.deterministic else int(np.random.randint(0, len(patches)))
        x_p, y_p = patches[pick]
        return {
            "image": torch.from_numpy(x_p.astype(np.float32)),
            "mask": torch.from_numpy(y_p.astype(np.int64)),
            "patient_id": pid,
        }
