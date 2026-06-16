r"""Inference script for onzediff — 2.5D segmentation with body-oval masking.

Usage
-----
python predict.py \
    --muscle R_gracilis \
    --ckpt_dir ~/onzediff_ckpts \
    --gt_base ~/myosegmenTUM \
    --out_dir ~/onzediff_segs \
    --ddim_steps 50

Body-oval masking:
  Input  — pixels outside the body oval are always zeroed before the U-Net
           (matches training preprocessing).
  Output — predicted mask is zeroed outside the body oval only when
           --body_oval_output is passed (off by default).
"""

from __future__ import annotations
import argparse
import glob
import os
import re
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from body_oval import get_body_mask
from dataset import GT_LABELS
from dataset import _norm
from dataset import discover_subjects
from diffusion import GaussianDiffusion
from unet import UNet


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="onzediff inference")
    p.add_argument("--muscle",      required=True, choices=list(GT_LABELS))
    p.add_argument("--ckpt_dir",    default=os.path.expanduser("~/onzediff_ckpts"))
    p.add_argument("--gt_base",     default=os.path.expanduser("~/myosegmenTUM"))
    p.add_argument("--out_dir",     default=os.path.expanduser("~/onzediff_segs"))
    p.add_argument("--img_size",    type=int, default=256)
    p.add_argument("--base_ch",     type=int, default=64)
    p.add_argument("--t_dim",       type=int, default=256)
    p.add_argument("--T",           type=int, default=1000)
    p.add_argument("--ddim_steps",  type=int, default=50)
    p.add_argument("--subjects",    nargs="*", default=None)
    p.add_argument(
        "--body_oval_output", default=False,
        action=argparse.BooleanOptionalAction,
        help="Zero the predicted mask outside the body oval (forces muscles inside body).",
    )
    return p.parse_args()


@torch.no_grad()
def segment_volume(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    water_path: str,
    ff_path: str | None,
    img_size: int,
    device: torch.device,
    ddim_steps: int,
    n_adjacent: int,
    body_oval_output: bool = False,
) -> np.ndarray:
    """Segment a single NIfTI water stack with 2.5D input and body-oval masking.

    Args:
        model: Trained U-Net denoising model.
        diffusion: GaussianDiffusion scheduler used for DDIM sampling.
        water_path: Path to the water NIfTI file.
        ff_path: Optional path to the fat-fraction NIfTI file.
        img_size: Spatial size to resize slices to before inference.
        device: Torch device to run inference on.
        ddim_steps: Number of DDIM denoising steps.
        n_adjacent: Number of adjacent slices stacked as input channels.
        body_oval_output: When True, zero the predicted mask outside the body
            oval, forcing predictions to stay inside the detected body region.

    Returns:
        Binary uint8 volume of shape (D, H_orig, W_orig).
    """
    w_arr = sitk.GetArrayFromImage(sitk.ReadImage(water_path)).astype(np.float32)
    D, H, W = w_arr.shape
    half = n_adjacent // 2

    ff_arr = None
    if ff_path is not None and os.path.exists(ff_path):
        ff_arr = sitk.GetArrayFromImage(sitk.ReadImage(ff_path)).astype(np.float32)

    pred_vol = np.zeros((D, H, W), dtype=np.uint8)

    for sl in range(D):
        # Body oval from the centre water slice (original resolution)
        body_mask = get_body_mask(_norm(w_arr[sl]))   # (H, W) bool

        # Build 2.5D channel stack with body masking
        channels = []
        for offset in range(-half, half + 1):
            si = int(np.clip(sl + offset, 0, D - 1))
            w_sl = _norm(w_arr[si])
            w_sl[~body_mask] = 0.0
            channels.append(torch.from_numpy(w_sl).unsqueeze(0))
            if ff_arr is not None:
                ff_sl = _norm(ff_arr[si])
                ff_sl[~body_mask] = 0.0
                channels.append(torch.from_numpy(ff_sl).unsqueeze(0))

        img_t = torch.cat(channels, dim=0) * 2.0 - 1.0   # (C, H, W) in [-1, 1]
        img_r = F.interpolate(
            img_t.unsqueeze(0), size=(img_size, img_size),
            mode="bilinear", align_corners=False,
        ).to(device)

        mask_pred = diffusion.ddim_sample(model, img_r, num_steps=ddim_steps)
        mask_r = F.interpolate(
            mask_pred, size=(H, W), mode="bilinear", align_corners=False,
        )

        pred_sl = (mask_r.squeeze().cpu().numpy() > 0.0).astype(np.uint8)
        if body_oval_output:
            pred_sl[~body_mask] = 0
        pred_vol[sl] = pred_sl

    return pred_vol


def main() -> None:
    """Load checkpoint and segment all water stacks."""
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    best_ckpt = os.path.join(args.ckpt_dir, f"{args.muscle}_best.pt")
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")

    ckpt = torch.load(best_ckpt, map_location=device)
    saved = ckpt.get("args", {})

    n_adjacent = saved.get("n_adjacent", 3)
    use_ff     = saved.get("use_ff",     False)
    base_ch    = 2 if use_ff else 1
    img_ch     = n_adjacent * base_ch

    model = UNet(
        img_ch=img_ch,
        base=saved.get("base_ch", args.base_ch),
        t_dim=saved.get("t_dim",  args.t_dim),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {best_ckpt}")
    print(f"  epoch={ckpt.get('epoch', '?')}  best_dice={ckpt.get('best_dice', 0.0):.4f}")
    print(f"  n_adjacent={n_adjacent}  use_ff={use_ff}  img_ch={img_ch}")

    diffusion = GaussianDiffusion(T=args.T, device=device)
    subjects  = args.subjects or discover_subjects(args.gt_base)

    for subj in sorted(subjects):
        subj_dir    = os.path.join(args.gt_base, subj)
        water_files = sorted(glob.glob(
            os.path.join(subj_dir, "ImageData", f"{subj}_WATER", f"{subj}_WATER_stack*.nii")
        ))
        for wpath in water_files:
            m = re.search(r"stack(\d+)\.nii$", wpath)
            if not m:
                continue
            stack   = m.group(1)
            stem    = f"{subj}_WATER_stack{stack}"
            out_npz = os.path.join(args.out_dir, f"{stem}_onzediff.npz")

            ff_path: str | None = None
            if use_ff:
                cand = os.path.join(
                    subj_dir, "ImageData",
                    f"{subj}_FATFRACTION",
                    f"{subj}_FATFRACTION_stack{stack}.nii",
                )
                ff_path = cand if os.path.exists(cand) else None

            print(f"Segmenting {stem} ...", end=" ", flush=True)
            pred = segment_volume(
                model, diffusion, wpath, ff_path,
                args.img_size, device, args.ddim_steps, n_adjacent,
                body_oval_output=args.body_oval_output,
            )
            print(f"{int(pred.sum()):,} positive voxels")

            existing: dict[str, np.ndarray] = {}
            if os.path.exists(out_npz):
                existing = dict(np.load(out_npz))
            existing[args.muscle] = pred
            np.savez_compressed(out_npz, **existing)

    print(f"\nDone.  Results in {args.out_dir}")


if __name__ == "__main__":
    main()
