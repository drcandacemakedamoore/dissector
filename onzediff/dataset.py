"""Multi-source 2.5D dataset with body-oval masking for onzediff.

Three data sources, unified into one training pool:
- myosegmenTUM: bilateral ground truth, but only Gracilis and Sartorius are
  individually resolvable -- its other two labels (Hamstrings, Quadriceps)
  are compound blobs covering several distinct muscles each, not usable for
  per-muscle training.
- Sheffield: single-leg (right thigh) DICOM acquisitions with 13
  individually-labeled muscles -- right-side data only.
- Asian: unilateral ground truth (only one side is labeled at all, the
  other side is background) with the same 13 individually-labeled muscles
  -- right-side data only.

Because Sheffield and Asian have no left-side ground truth, only the
right-side Gracilis/Sartorius models can draw on myosegmenTUM *and* the
other two sources; left-side Gracilis/Sartorius trains on myosegmenTUM
alone. Every other muscle (rectus_femoris, vastus_*, biceps_femoris,
semitendinosus, semimembranosus, adductor_*, gluteus_maximus) is only ever
trainable as a single right-side model, sourced from Sheffield and/or Asian
-- myosegmenTUM cannot contribute to these at all.

Key differences from medsegdiff/dataset.py:
- n_adjacent adjacent slices are stacked as input channels (2.5D context).
- Body oval is computed from the centre water slice and applied to every
  channel when use_body_oval=True: pixels outside the body are zeroed.
"""

from __future__ import annotations
import glob
import os
import random
import re
import numpy as np
import pydicom
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from body_oval import get_body_mask
from torch.utils.data import Dataset

LabelSpec = int  # or list[int], OR'd together -- see _match_labels

# myosegmenTUM combined_gt label convention. Only Gracilis and Sartorius are
# individually resolvable -- this is the only source with a left/right split.
MYOSEGMENTUM_GT_LABELS: dict[str, int] = {
    "R_gracilis":  5,
    "L_gracilis":  1,
    "R_sartorius": 8,
    "L_sartorius": 4,
}

# Sheffield DICOM GT: 20440203/Aug_N_segmentations.dcm, single-leg (right)
# acquisition, labels 1-37. biceps_femoris is split into long/short heads in
# this scheme -- OR'd together to match Asian's single combined label.
SHEFFIELD_GT_LABEL: dict[str, LabelSpec] = {
    "adductor_brevis":     1,
    "adductor_longus":     2,
    "adductor_magnus":     3,
    "biceps_femoris":      [4, 5],   # short head + long head
    "gracilis":            16,
    "rectus_femoris":      27,
    "sartorius":           28,
    "semimembranosus":     29,
    "semitendinosus":      30,
    "vastus_intermedius":  35,
    "vastus_lateralis":    36,
    "vastus_medialis":     37,
    # no gluteus_maximus in the Sheffield thigh scheme
}

# Asian NIfTI GT: {subject}/Thigh/mask_muscles.nii.gz, unilateral (right-side
# only; the left side is background in every subject), labels 1-13.
ASIAN_GT_LABEL: dict[str, LabelSpec] = {
    "rectus_femoris":      1,
    "vastus_lateralis":    2,
    "vastus_intermedius":  3,
    "vastus_medialis":     4,
    "sartorius":           5,
    "gracilis":            6,
    "biceps_femoris":      7,
    "semitendinosus":      8,
    "semimembranosus":     9,
    "adductor_brevis":     10,
    "adductor_longus":     11,
    "adductor_magnus":     12,
    "gluteus_maximus":     13,
}

# Every muscle key accepted by --muscle in train.py / predict.py.
# Gracilis/Sartorius are side-specific (R_/L_ prefix, myosegmenTUM-capable);
# everything else is inherently right-side-only (Sheffield/Asian only), so
# no side prefix is used for those.
ALL_MUSCLE_CHOICES: list[str] = sorted(
    set(MYOSEGMENTUM_GT_LABELS)
    | (set(SHEFFIELD_GT_LABEL) | set(ASIAN_GT_LABEL)) - {"gracilis", "sartorius"}
)


def _match_labels(arr: np.ndarray, label_spec: LabelSpec) -> np.ndarray:
    """Boolean mask of arr matching label_spec (an int, or list[int] OR'd together)."""
    if isinstance(label_spec, (list, tuple)):
        out = np.zeros(arr.shape, dtype=bool)
        for lbl in label_spec:
            out |= (arr == lbl)
        return out
    return arr == label_spec


def _norm(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile-clip then scale to [0, 1]."""
    lo_v = np.percentile(arr, lo)
    hi_v = np.percentile(arr, hi)
    arr = np.clip(arr, lo_v, hi_v)
    rng = hi_v - lo_v
    return ((arr - lo_v) / rng).astype(np.float32) if rng > 0 else np.zeros_like(arr, dtype=np.float32)


def discover_subjects(gt_base: str) -> list[str]:
    """Return sorted list of myosegmenTUM subject IDs found under gt_base."""
    return sorted(d for d in os.listdir(gt_base) if os.path.isdir(os.path.join(gt_base, d)))


def discover_asian_subjects(asian_root: str) -> list[str]:
    """Return sorted list of Asian-dataset subject IDs with a Thigh GT mask.

    asian_root is the MRI_data_asian/MRI_data directory (one subdir per
    subject, e.g. '01', '02', ...).
    """
    out = []
    for d in sorted(os.listdir(asian_root)):
        subj_dir = os.path.join(asian_root, d)
        if not os.path.isdir(subj_dir):
            continue
        if os.path.exists(os.path.join(subj_dir, "Thigh", "mask_muscles.nii.gz")):
            out.append(d)
    return out


def train_val_split(
    items: list[str], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split a list of subject IDs (or Sheffield Aug_N indices) into train/val."""
    rng = random.Random(seed)
    s = list(items)
    rng.shuffle(s)
    n_val = max(1, round(len(s) * val_fraction))
    return s[n_val:], s[:n_val]


class _Sample:
    """One trainable 2.5D slice, tagged with which loader it needs."""

    __slots__ = ("source", "water_path", "ff_path", "gt_path", "slice_idx", "gt_label")

    def __init__(self, source, water_path, ff_path, gt_path, slice_idx, gt_label: LabelSpec):
        self.source     = source       # 'myosegmentum' | 'sheffield' | 'asian'
        self.water_path = water_path
        self.ff_path    = ff_path      # None if no fat-fraction channel available
        self.gt_path    = gt_path
        self.slice_idx  = slice_idx
        self.gt_label   = gt_label     # int or list[int] to match in this sample's GT volume


def _read_sheffield_gt_volume(gt_path: str) -> np.ndarray:
    """Decode a Sheffield Aug_N_segmentations.dcm into an integer label volume."""
    ds  = pydicom.dcmread(gt_path)
    raw = ds.pixel_array.astype(np.float32)
    if raw.ndim == 2:
        raw = raw[np.newaxis]
    labeled = np.round(raw * 37.0 / 255.0).astype(np.int32)
    labeled[raw == 0] = 0
    return np.clip(labeled, 0, 37)


def discover_myosegmentum_samples(
    gt_base: str, subjects: list[str], muscle: str, min_fg_voxels: int = 50,
) -> list[_Sample]:
    """muscle is a side-specific key, e.g. 'R_gracilis' -- this is the only
    source with a left/right split, and only Gracilis/Sartorius are usable."""
    gt_label = MYOSEGMENTUM_GT_LABELS[muscle]
    samples: list[_Sample] = []
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
            ff_path = os.path.join(
                subj_dir, "ImageData", f"{subj}_FATFRACTION", f"{subj}_FATFRACTION_stack{stack}.nii",
            )
            ff_path = ff_path if os.path.exists(ff_path) else None
            gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
            for i in range(gt_arr.shape[0]):
                if int(_match_labels(gt_arr[i], gt_label).sum()) >= min_fg_voxels:
                    samples.append(_Sample("myosegmentum", wpath, ff_path, gt_path, i, gt_label))
    return samples


def discover_sheffield_samples(
    dcm_dir: str, gt_dir: str, base_muscle: str,
    indices: list[str] | None = None, min_fg_voxels: int = 50,
) -> list[_Sample]:
    """base_muscle must be a key of SHEFFIELD_GT_LABEL -- Sheffield is
    single-leg (right thigh only), so there is no L/R choice here.

    dcm_dir: directory of Aug_N.dcm water images (e.g. ~/sheffeld/20440164).
    gt_dir:  directory of Aug_N_segmentations.dcm GT (e.g. ~/sheffeld/20440203).
    indices: optional subset of Aug_N indices (as strings) to include, for
        train/val splitting -- if None, all available indices are used.
    """
    gt_label = SHEFFIELD_GT_LABEL[base_muscle]
    samples: list[_Sample] = []
    dcm_files = sorted(
        f for f in glob.glob(os.path.join(dcm_dir, "Aug_*.dcm"))
        if "_segmentations" not in f
    )
    for wpath in dcm_files:
        m = re.search(r"Aug_(\d+)\.dcm$", wpath)
        if not m:
            continue
        idx = m.group(1)
        if indices is not None and idx not in indices:
            continue
        gt_path = os.path.join(gt_dir, f"Aug_{idx}_segmentations.dcm")
        if not os.path.exists(gt_path):
            continue
        gt_arr = _read_sheffield_gt_volume(gt_path)
        for i in range(gt_arr.shape[0]):
            if int(_match_labels(gt_arr[i], gt_label).sum()) >= min_fg_voxels:
                samples.append(_Sample("sheffield", wpath, None, gt_path, i, gt_label))
    return samples


def discover_asian_samples(
    asian_root: str, subjects: list[str], base_muscle: str, min_fg_voxels: int = 50,
) -> list[_Sample]:
    """base_muscle must be a key of ASIAN_GT_LABEL -- the Asian GT is
    unilateral (right-side only), so there is no L/R choice here either."""
    gt_label = ASIAN_GT_LABEL[base_muscle]
    samples: list[_Sample] = []
    for subj in subjects:
        thigh_dir  = os.path.join(asian_root, subj, "Thigh")
        water_path = os.path.join(thigh_dir, "Water.nii.gz")
        gt_path    = os.path.join(thigh_dir, "mask_muscles.nii.gz")
        if not (os.path.exists(water_path) and os.path.exists(gt_path)):
            continue
        ff_path = os.path.join(thigh_dir, "Fat.nii.gz")
        ff_path = ff_path if os.path.exists(ff_path) else None
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        for i in range(gt_arr.shape[0]):
            if int(_match_labels(gt_arr[i], gt_label).sum()) >= min_fg_voxels:
                samples.append(_Sample("asian", water_path, ff_path, gt_path, i, gt_label))
    return samples


def _read_volume_triplet(sample: _Sample) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return (water_vol, ff_vol_or_None, gt_vol), each (D, H, W)."""
    if sample.source == "sheffield":
        ds = pydicom.dcmread(sample.water_path)
        water_vol = ds.pixel_array.astype(np.float32)
        if water_vol.ndim == 2:
            water_vol = water_vol[np.newaxis]
        gt_vol = _read_sheffield_gt_volume(sample.gt_path)
        return water_vol, None, gt_vol

    # myosegmenTUM and Asian both use plain NIfTI/.mha via SimpleITK.
    water_vol = sitk.GetArrayFromImage(sitk.ReadImage(sample.water_path)).astype(np.float32)
    ff_vol = None
    if sample.ff_path is not None:
        ff_vol = sitk.GetArrayFromImage(sitk.ReadImage(sample.ff_path)).astype(np.float32)
    gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(sample.gt_path))
    return water_vol, ff_vol, gt_vol


class DixonThighDataset(Dataset):
    """2.5D dataset: stacks n_adjacent axial slices as channels.

    Takes a pre-built list of _Sample (see discover_*_samples above) rather
    than discovering its own files, so a single instance can mix slices from
    myosegmenTUM, Sheffield, and Asian without needing to know about any of
    their directory layouts itself.

    For each centre slice sl:
      - Builds body oval from the normalised centre water slice (skipped
        entirely when use_body_oval=False).
      - Stacks [sl - half, ..., sl, ..., sl + half] water channels
        (and fat-fraction channels if use_ff=True -- zero-filled for samples
        whose source has no fat-fraction data, e.g. Sheffield), zeroing
        pixels outside the body oval in every channel when enabled.
      - img shape:  (n_adjacent * base_ch, img_size, img_size) in [-1, 1]
      - mask shape: (1, img_size, img_size) in {-1, +1}

    Edge slices are clamped (repeat-pad): sl=-1 -> sl=0, sl=D -> sl=D-1.
    """

    def __init__(
        self,
        samples: list[_Sample],
        img_size: int = 256,
        use_ff: bool = True,
        augment: bool = True,
        n_adjacent: int = 3,
        use_body_oval: bool = True,
    ) -> None:
        self.samples       = samples
        self.img_size      = img_size
        self.use_ff        = use_ff
        self.augment       = augment
        self.n_adjacent    = n_adjacent
        self.half          = n_adjacent // 2
        self.use_body_oval = use_body_oval

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        w_vol, ff_vol, gt_vol = _read_volume_triplet(sample)
        D  = w_vol.shape[0]
        sl = sample.slice_idx

        # Body oval from centre slice -- fallback to all-True if detection
        # fails, or skipped entirely if use_body_oval=False.
        body_mask = get_body_mask(_norm(w_vol[sl])) if self.use_body_oval else None

        # Build 2.5D channel stack
        channels = []
        for offset in range(-self.half, self.half + 1):
            si = int(np.clip(sl + offset, 0, D - 1))
            w_sl = _norm(w_vol[si])
            if body_mask is not None:
                w_sl[~body_mask] = 0.0
            channels.append(w_sl)
            if self.use_ff:
                if ff_vol is not None:
                    ff_sl = _norm(ff_vol[si])
                    if body_mask is not None:
                        ff_sl[~body_mask] = 0.0
                else:
                    # Source has no fat-fraction data (e.g. Sheffield) --
                    # zero-fill so every sample in a batch has the same
                    # channel count regardless of source.
                    ff_sl = np.zeros_like(w_sl)
                channels.append(ff_sl)

        img  = torch.from_numpy(np.stack(channels, axis=0))          # (C, H, W)
        mask = torch.from_numpy(
            _match_labels(gt_vol[sl], sample.gt_label).astype(np.float32)
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
        """Total input channels: n_adjacent x base_channels."""
        return self.n_adjacent * (2 if self.use_ff else 1)
