"""myosegmenTUM dataset for diffusion-based thigh muscle segmentation.

Each sample is a 2D axial slice from a Dixon water stack, optionally paired
with the corresponding fat-fraction slice as a second input channel.
The target mask is a binary {-1, +1} array for a single muscle.
"""

from __future__ import annotations
import glob
import os
import random
import re
import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# GT label index for each muscle in combined_gt_stack*.mha
GT_LABELS: dict[str, int] = {
    "R_gracilis":  5,
    "L_gracilis":  1,
    "R_sartorius": 8,
    "L_sartorius": 4,
}


def _norm(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile-clip then scale to [0, 1]."""
    lo_v = np.percentile(arr, lo)
    hi_v = np.percentile(arr, hi)
    arr = np.clip(arr, lo_v, hi_v)
    rng = hi_v - lo_v
    return ((arr - lo_v) / rng).astype(np.float32) if rng > 0 else np.zeros_like(arr, dtype=np.float32)


def discover_subjects(gt_base: str) -> list[str]:
    """Return sorted list of subject IDs found under gt_base."""
    return sorted(
        d for d in os.listdir(gt_base)
        if os.path.isdir(os.path.join(gt_base, d))
    )


def train_val_split(subjects: list[str], val_fraction: float = 0.2, seed: int = 42
                    ) -> tuple[list[str], list[str]]:
    """Split subject list into train and validation sets by subject (not by slice).

    Args:
        subjects: sorted list of subject IDs.
        val_fraction: fraction of subjects to use for validation.
        seed: random seed for reproducibility.

    Returns:
        Tuple of (train_subjects, val_subjects).
    """
    rng = random.Random(seed)
    s = list(subjects)
    rng.shuffle(s)
    n_val = max(1, round(len(s) * val_fraction))
    return s[n_val:], s[:n_val]


class DixonThighDataset(Dataset):
    """Dataset of 2D axial slices for diffusion-based muscle segmentation.

    Yields (img, mask) pairs where:
    img  — (C, img_size, img_size) float32 in [-1, 1]; C=1 water or C=2 water+FF.
    mask — (1, img_size, img_size) float32 in {-1, +1}.
    """

    def __init__(
        self,
        gt_base: str,
        muscle: str,
        subjects: list[str],
        img_size: int = 256,
        use_ff: bool = True,
        augment: bool = True,
        min_fg_voxels: int = 50,
    ) -> None:
        if muscle not in GT_LABELS:
            msg = f'Unknown muscle "{muscle}". Choose from {list(GT_LABELS)}'
            raise ValueError(msg)
        self.gt_label = GT_LABELS[muscle]
        self.img_size = img_size
        self.use_ff = use_ff
        self.augment = augment

        # Each entry: (water_path, ff_path_or_None, gt_path, slice_idx)
        self.samples: list[tuple[str, str | None, str, int]] = []
        self._discover(gt_base, subjects, min_fg_voxels)

    # ------------------------------------------------------------------
    def _discover(self, gt_base: str, subjects: list[str], min_fg: int) -> None:
        for subj in subjects:
            subj_dir = os.path.join(gt_base, subj)
            water_glob = os.path.join(
                subj_dir, "ImageData", f"{subj}_WATER", f"{subj}_WATER_stack*.nii"
            )
            for wpath in sorted(glob.glob(water_glob)):
                m = re.search(r"stack(\d+)\.nii$", wpath)
                if not m:
                    continue
                stack = m.group(1)
                gt_path = os.path.join(
                    subj_dir, "SegmentationMasks", f"combined_gt_stack{stack}.mha"
                )
                if not os.path.exists(gt_path):
                    continue

                ff_path: str | None = None
                if self.use_ff:
                    cand = os.path.join(
                        subj_dir, "ImageData",
                        f"{subj}_FATFRACTION",
                        f"{subj}_FATFRACTION_stack{stack}.nii",
                    )
                    if os.path.exists(cand):
                        ff_path = cand

                gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
                for i in range(gt_arr.shape[0]):
                    if int((gt_arr[i] == self.gt_label).sum()) >= min_fg:
                        self.samples.append((wpath, ff_path, gt_path, i))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wpath, ff_path, gt_path, sl = self.samples[idx]

        w_arr = sitk.GetArrayFromImage(sitk.ReadImage(wpath)).astype(np.float32)
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))

        w_sl = _norm(w_arr[sl])
        channels = [w_sl]

        if ff_path is not None:
            ff_arr = sitk.GetArrayFromImage(sitk.ReadImage(ff_path)).astype(np.float32)
            channels.append(_norm(ff_arr[sl]))

        img = torch.from_numpy(np.stack(channels, axis=0))          # (C, H, W) in [0,1]
        mask = torch.from_numpy(
            (gt_arr[sl] == self.gt_label).astype(np.float32)
        ).unsqueeze(0)                                                # (1, H, W) in {0,1}

        # Resize
        img = TF.resize(img, [self.img_size, self.img_size], antialias=True)
        mask = TF.resize(
            mask, [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        # Augmentation (applied identically to image and mask)
        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.uniform(-15.0, 15.0)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Scale to [-1, 1] for diffusion
        img = img * 2.0 - 1.0
        mask = mask * 2.0 - 1.0   # {0,1} → {-1,+1}

        return img, mask

    @property
    def n_image_channels(self) -> int:
        """Number of image channels inferred from the first sample."""
        if not self.samples:
            return 1
        _, ff_path, _, _ = self.samples[0]
        return 2 if (self.use_ff and ff_path is not None) else 1
