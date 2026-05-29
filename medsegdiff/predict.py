"""
Inference script: load a trained checkpoint and segment a full 3D volume.

Usage
-----
python predict.py \\
    --muscle R_gracilis \\
    --ckpt_dir ~/medsegdiff_ckpts \\
    --gt_base ~/myosegmenTUM \\
    --out_dir ~/medsegdiff_segs_water \\
    --img_size 256 \\
    --ddim_steps 50 \\
    --use_ff

Output: one NPZ per stack with keys = muscle names, arrays = (D, H_orig, W_orig) uint8.
If you run predict.py for multiple muscles they accumulate into the same NPZ files.
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

from dataset import GT_LABELS, _norm, discover_subjects
from unet import UNet
from diffusion import GaussianDiffusion


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--muscle',     required=True, choices=list(GT_LABELS))
    p.add_argument('--ckpt_dir',   default=os.path.expanduser('~/medsegdiff_ckpts'))
    p.add_argument('--gt_base',    default=os.path.expanduser('~/myosegmenTUM'))
    p.add_argument('--out_dir',    default=os.path.expanduser('~/medsegdiff_segs_water'))
    p.add_argument('--img_size',   type=int,   default=256)
    p.add_argument('--base_ch',    type=int,   default=64)
    p.add_argument('--t_dim',      type=int,   default=256)
    p.add_argument('--T',          type=int,   default=1000)
    p.add_argument('--ddim_steps', type=int,   default=50)
    p.add_argument('--use_ff',     action='store_true')
    p.add_argument('--subjects',   nargs='*',  default=None,
                   help='Limit to these subject IDs; default = all')
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
) -> np.ndarray:
    """
    Segment a single NIfTI water stack.
    Returns binary uint8 volume (D, H_orig, W_orig).
    """
    w_arr = sitk.GetArrayFromImage(sitk.ReadImage(water_path)).astype(np.float32)
    D, H, W = w_arr.shape

    ff_arr = None
    if ff_path is not None:
        ff_arr = sitk.GetArrayFromImage(sitk.ReadImage(ff_path)).astype(np.float32)

    pred_vol = np.zeros((D, H, W), dtype=np.uint8)

    for sl in range(D):
        w_sl = _norm(w_arr[sl])
        channels = [torch.from_numpy(w_sl).unsqueeze(0)]   # (1, H, W)
        if ff_arr is not None:
            channels.append(torch.from_numpy(_norm(ff_arr[sl])).unsqueeze(0))

        img_t = torch.stack(channels, dim=0).squeeze(0)    # (C, H, W)  ← was list of (1,H,W)
        # correct shape: stack and squeeze introduced an extra dim; rebuild:
        img_t = torch.cat(channels, dim=0)                 # (C, H, W) in [0,1]
        img_t = img_t * 2.0 - 1.0                          # → [-1, 1]

        # Resize to model input size
        img_r = F.interpolate(
            img_t.unsqueeze(0), size=(img_size, img_size),
            mode='bilinear', align_corners=False,
        )  # (1, C, img_size, img_size)

        img_r = img_r.to(device)
        mask_pred = diffusion.ddim_sample(model, img_r, num_steps=ddim_steps)
        # mask_pred: (1, 1, img_size, img_size) in [-1, 1]

        # Resize back to original spatial resolution
        mask_r = F.interpolate(
            mask_pred, size=(H, W), mode='bilinear', align_corners=False,
        )  # (1, 1, H, W)

        pred_vol[sl] = (mask_r.squeeze().cpu().numpy() > 0.0).astype(np.uint8)

    return pred_vol


def main() -> None:
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    best_ckpt = os.path.join(args.ckpt_dir, f'{args.muscle}_best.pt')
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {best_ckpt}')

    ckpt = torch.load(best_ckpt, map_location=device)
    saved_args = ckpt.get('args', {})
    img_ch = 2 if (args.use_ff or saved_args.get('use_ff', False)) else 1

    model = UNet(
        img_ch=img_ch,
        base=saved_args.get('base_ch', args.base_ch),
        t_dim=saved_args.get('t_dim',  args.t_dim),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded {best_ckpt}  (epoch {ckpt.get("epoch", "?")},'
          f' best Dice {ckpt.get("best_dice", 0.0):.4f})')

    diffusion = GaussianDiffusion(T=args.T, device=device)

    # ── discover stacks ───────────────────────────────────────────────────────
    subjects = args.subjects or discover_subjects(args.gt_base)

    for subj in sorted(subjects):
        subj_dir = os.path.join(args.gt_base, subj)
        water_files = sorted(glob.glob(
            os.path.join(subj_dir, 'ImageData', f'{subj}_WATER', f'{subj}_WATER_stack*.nii')
        ))
        for wpath in water_files:
            m = re.search(r'stack(\d+)\.nii$', wpath)
            if not m:
                continue
            stack = m.group(1)
            stem  = f'{subj}_WATER_stack{stack}'
            out_npz = os.path.join(args.out_dir, f'{stem}_medsegdiff.npz')

            ff_path: str | None = None
            if args.use_ff:
                cand = os.path.join(
                    subj_dir, 'ImageData',
                    f'{subj}_FATFRACTION',
                    f'{subj}_FATFRACTION_stack{stack}.nii',
                )
                ff_path = cand if os.path.exists(cand) else None

            print(f'Segmenting {stem} ...', end=' ', flush=True)
            pred = segment_volume(
                model, diffusion, wpath, ff_path,
                args.img_size, device, args.ddim_steps,
            )
            voxels = int(pred.sum())
            print(f'{voxels:,} positive voxels')

            # Accumulate into NPZ (other muscles may have been predicted already)
            existing: dict[str, np.ndarray] = {}
            if os.path.exists(out_npz):
                existing = dict(np.load(out_npz))
            existing[args.muscle] = pred
            np.savez_compressed(out_npz, **existing)

    print(f'\nDone.  Results in {args.out_dir}')


if __name__ == '__main__':
    main()
