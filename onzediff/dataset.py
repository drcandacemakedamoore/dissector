"""myosegmenTUM 2.5D dataset with body-oval masking for onzediff.

Key differences from medsegdiff/dataset.py:
- n_adjacent adjacent slices are stacked as input channels (2.5D context).
- Body oval is computed from the centre water slice and applied to every
  channel: pixels outside the body are set to zero before training.
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

from body_oval import get_body_mask

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
    return sorted(d for d in os.listdir(gt_base) if os.path.isdir(os.path.join(gt_base, d)))


def train_val_split(
    subjects: list[str], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split subjects into train / val by subject (not by slice)."""
    rng = random.Random(seed)
    s = list(subjects)
    rng.shuffle(s)
    n_val = max(1, round(len(s) * val_fraction))
    return s[n_val:], s[:n_val]


class DixonThighDataset(Dataset):
    """2.5D dataset: stacks n_adjacent axial slices as channels.

    For each centre slice sl:
      - Builds body oval from the normalised centre water slice.
      - Stacks [sl - half, ..., sl, ..., sl + half] water channels
        (and fat-fraction channels if use_ff=True), zeroing pixels outside
        the body oval in every channel.
      - img shape:  (n_adjacent * base_ch, img_size, img_size) in [-1, 1]
      - mask shape: (1, img_size, img_size) in {-1, +1}

    Edge slices are clamped (repeat-pad): sl=-1 → sl=0, sl=D → sl=D-1.
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
        n_adjacent: int = 3,
    ) -> None:
        if muscle not in GT_LABELS:
            msg = f'Unknown muscle "{muscle}". Choose from {list(GT_LABELS)}'
            raise ValueError(msg)
        self.gt_label = GT_LABELS[muscle]
        self.img_size = img_size
        self.use_ff = use_ff
        self.augment = augment
        self.n_adjacent = n_adjacent
        self.half = n_adjacent // 2

        self.samples: list[tuple[str, str | None, str, int]] = []
        self._discover(gt_base, subjects, min_fg_voxels)

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wpath, ff_path, gt_path, sl = self.samples[idx]

        w_vol  = sitk.GetArrayFromImage(sitk.ReadImage(wpath)).astype(np.float32)
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        D = w_vol.shape[0]

        ff_vol = None
        if ff_path is not None:
            ff_vol = sitk.GetArrayFromImage(sitk.ReadImage(ff_path)).astype(np.float32)

        # Body oval from centre slice — fallback to all-True if detection fails
        body_mask = get_body_mask(_norm(w_vol[sl]))   # (H, W) bool

        # Build 2.5D channel stack
        channels = []
        for offset in range(-self.half, self.half + 1):
            si = int(np.clip(sl + offset, 0, D - 1))
            w_sl = _norm(w_vol[si])
            w_sl[~body_mask] = 0.0
            channels.append(w_sl)
            if ff_vol is not None:
                ff_sl = _norm(ff_vol[si])
                ff_sl[~body_mask] = 0.0
                channels.append(ff_sl)

        img  = torch.from_numpy(np.stack(channels, axis=0))          # (C, H, W)
        mask = torch.from_numpy(
            (gt_arr[sl] == self.gt_label).astype(np.float32)
        ).unsqueeze(0)                                                 # (1, H, W)

        img  = TF.resize(img,  [self.img_size, self.img_size], antialias=True)
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.uniform(-15.0, 15.0)
            img  = TF.rotate(img,  angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        img  = img  * 2.0 - 1.0
        mask = mask * 2.0 - 1.0

        return img, mask

    @property
    def n_image_channels(self) -> int:
        """Total input channels: n_adjacent × base_channels."""
        if not self.samples:
            return self.n_adjacent
        _, ff_path, _, _ = self.samples[0]
        base_ch = 2 if (self.use_ff and ff_path is not None) else 1
        return self.n_adjacent * base_ch
