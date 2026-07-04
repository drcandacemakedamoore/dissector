r"""Training script for onzediff -- 2.5D diffusion segmentation with body-oval masking.

Combines up to three data sources. Gracilis and Sartorius are the only
muscles myosegmenTUM resolves individually, so only those two accept a
side prefix (R_/L_) and can draw on myosegmenTUM + Sheffield + Asian for
the right side (myosegmenTUM alone for the left side, since Sheffield and
Asian have no left-side ground truth -- Sheffield is a single right-leg
acquisition, Asian's GT is unilateral).

Every other muscle (rectus_femoris, vastus_lateralis, vastus_intermedius,
vastus_medialis, biceps_femoris, semitendinosus, semimembranosus,
adductor_brevis, adductor_longus, adductor_magnus, gluteus_maximus) is
right-side-only by construction and sourced from Sheffield and/or Asian --
myosegmenTUM cannot contribute to these at all (its GT only has compound
Hamstrings/Quadriceps blobs, not the individual muscles).

Usage
-----
# Gracilis, right side -- combines all three datasets
python train.py \
    --muscle R_gracilis \
    --gt_base ~/myosegmenTUM \
    --sheffield_dcm_dir ~/sheffeld/20440164 \
    --sheffield_gt_dir ~/sheffeld/20440203 \
    --asian_root ~/MRI_data_asian/MRI_data \
    --out_dir ~/onzediff_ckpts \
    --epochs 300 --batch_size 8 --n_adjacent 3 --use_ff

# Left-side Gracilis only ever uses myosegmenTUM, even if the other
# directories are passed -- they're simply ignored with a printed note.
python train.py --muscle L_gracilis --gt_base ~/myosegmenTUM --out_dir ~/onzediff_ckpts

# A Sheffield/Asian-only muscle (no myosegmenTUM data exists for it):
python train.py --muscle vastus_lateralis \
    --sheffield_dcm_dir ~/sheffeld/20440164 --sheffield_gt_dir ~/sheffeld/20440203 \
    --asian_root ~/MRI_data_asian/MRI_data --out_dir ~/onzediff_ckpts

# Train the unmasked (no body-oval) variant of the same muscle:
python train.py --muscle R_gracilis --gt_base ~/myosegmenTUM --out_dir ~/onzediff_ckpts --no-use_body_oval
"""

from __future__ import annotations
import argparse
import glob
import os
import re
import time
import torch
from dataset import ALL_MUSCLE_CHOICES
from dataset import ASIAN_GT_LABEL
from dataset import discover_asian_samples
from dataset import discover_asian_subjects
from dataset import discover_myosegmentum_samples
from dataset import discover_sheffield_samples
from dataset import discover_subjects
from dataset import DixonThighDataset
from dataset import MYOSEGMENTUM_GT_LABELS
from dataset import SHEFFIELD_GT_LABEL
from dataset import train_val_split
from diffusion import GaussianDiffusion
from torch import optim
from torch.utils.data import DataLoader
from unet import UNet


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train onzediff on myosegmenTUM + Sheffield + Asian")
    p.add_argument("--muscle",      required=True, choices=ALL_MUSCLE_CHOICES)
    p.add_argument("--gt_base",     default=os.path.expanduser("~/myosegmenTUM"))
    p.add_argument("--sheffield_dcm_dir", default=None,
                   help="Sheffield Aug_N.dcm directory, e.g. ~/sheffeld/20440164")
    p.add_argument("--sheffield_gt_dir", default=None,
                   help="Sheffield Aug_N_segmentations.dcm directory, e.g. ~/sheffeld/20440203")
    p.add_argument("--asian_root",  default=None,
                   help="MRI_data_asian/MRI_data directory")
    p.add_argument("--out_dir",     default=os.path.expanduser("~/onzediff_ckpts"))
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--img_size",    type=int,   default=256)
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--t_dim",       type=int,   default=256)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--n_adjacent",  type=int,   default=3,
                   help="Number of adjacent slices to stack as channels (must be odd)")
    p.add_argument("--use_ff",      action="store_true",
                   help="Include fat-fraction as extra channel for each adjacent slice "
                        "(zero-filled for sources with no fat-fraction data, e.g. Sheffield)")
    p.add_argument(
        "--use_body_oval", default=True, action=argparse.BooleanOptionalAction,
        help="Zero pixels outside the body oval before training (default: on). "
             "Pass --no-use_body_oval to train the unmasked variant.",
    )
    p.add_argument("--min_fg_voxels", type=int, default=50,
                   help="Minimum foreground voxels for a slice to be included")
    p.add_argument("--val_frac",    type=float, default=0.2)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--save_every",  type=int,   default=25)
    p.add_argument("--ddim_steps",  type=int,   default=50)
    p.add_argument("--val_every",   type=int,   default=25)
    return p.parse_args()


def validate(model, diffusion, loader, device, ddim_steps):
    """Run DDIM on validation loader and return mean Dice."""
    model.eval()
    scores = []
    for img_cpu, mask_cpu in loader:
        img_d  = img_cpu.to(device)
        mask_d = mask_cpu.to(device)
        pred = diffusion.ddim_sample(model, img_d, num_steps=ddim_steps)
        scores.append(diffusion.dice((pred > 0).float(), (mask_d > 0).float()))
    model.train()
    return float(sum(scores) / max(len(scores), 1))


def build_samples(args) -> tuple[list, list]:
    """Discover and combine train/val samples from every available source."""
    muscle = args.muscle
    train_samples: list = []
    val_samples: list   = []

    if muscle in MYOSEGMENTUM_GT_LABELS:
        # Gracilis or Sartorius -- myosegmenTUM is usable, and is the only
        # source for the left side.
        is_right    = muscle.startswith("R_")
        base_muscle = muscle[2:]  # strip 'R_' / 'L_'

        my_subjects = discover_subjects(args.gt_base)
        my_train_subj, my_val_subj = train_val_split(my_subjects, val_fraction=args.val_frac)
        my_train = discover_myosegmentum_samples(args.gt_base, my_train_subj, muscle, args.min_fg_voxels)
        my_val   = discover_myosegmentum_samples(args.gt_base, my_val_subj,   muscle, args.min_fg_voxels)
        train_samples += my_train
        val_samples   += my_val
        print(f"myosegmenTUM : {len(my_train_subj)} train / {len(my_val_subj)} val subjects"
              f" -> {len(my_train)} / {len(my_val)} slices")
    else:
        # Every other muscle: myosegmenTUM's GT only has compound
        # Hamstrings/Quadriceps blobs, not this muscle individually.
        is_right    = True
        base_muscle = muscle
        print("myosegmenTUM : skipped (not individually resolvable in myosegmenTUM GT)")

    if args.sheffield_dcm_dir and args.sheffield_gt_dir:
        if not is_right:
            print("Sheffield    : skipped (no left-side ground truth available)")
        elif base_muscle not in SHEFFIELD_GT_LABEL:
            print(f"Sheffield    : skipped ({base_muscle!r} not in Sheffield's label scheme)")
        else:
            sheff_indices = sorted(
                re.search(r"Aug_(\d+)\.dcm$", f).group(1)
                for f in glob.glob(os.path.join(args.sheffield_dcm_dir, "Aug_*.dcm"))
                if "_segmentations" not in f
            )
            sheff_train_idx, sheff_val_idx = train_val_split(sheff_indices, val_fraction=args.val_frac)
            sh_train = discover_sheffield_samples(
                args.sheffield_dcm_dir, args.sheffield_gt_dir, base_muscle,
                sheff_train_idx, args.min_fg_voxels,
            )
            sh_val = discover_sheffield_samples(
                args.sheffield_dcm_dir, args.sheffield_gt_dir, base_muscle,
                sheff_val_idx, args.min_fg_voxels,
            )
            train_samples += sh_train
            val_samples   += sh_val
            print(f"Sheffield    : {len(sheff_train_idx)} train / {len(sheff_val_idx)} val volumes"
                  f" -> {len(sh_train)} / {len(sh_val)} slices")

    if args.asian_root:
        if not is_right:
            print("Asian        : skipped (no left-side ground truth available)")
        elif base_muscle not in ASIAN_GT_LABEL:
            print(f"Asian        : skipped ({base_muscle!r} not in Asian's label scheme)")
        else:
            asian_subjects = discover_asian_subjects(args.asian_root)
            asian_train_subj, asian_val_subj = train_val_split(asian_subjects, val_fraction=args.val_frac)
            as_train = discover_asian_samples(args.asian_root, asian_train_subj, base_muscle, args.min_fg_voxels)
            as_val   = discover_asian_samples(args.asian_root, asian_val_subj,   base_muscle, args.min_fg_voxels)
            train_samples += as_train
            val_samples   += as_val
            print(f"Asian        : {len(asian_train_subj)} train / {len(asian_val_subj)} val subjects"
                  f" -> {len(as_train)} / {len(as_val)} slices")

    return train_samples, val_samples


def main() -> None:
    """Train a binary onzediff model for one muscle, optionally combining datasets."""
    args = get_args()
    if args.n_adjacent % 2 == 0:
        raise ValueError("--n_adjacent must be odd (e.g. 1, 3, 5)")
    if args.muscle not in MYOSEGMENTUM_GT_LABELS and not (args.sheffield_dcm_dir or args.asian_root):
        raise ValueError(
            f"--muscle {args.muscle!r} has no myosegmenTUM data -- pass "
            "--sheffield_dcm_dir/--sheffield_gt_dir and/or --asian_root."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Muscle       : {args.muscle}")
    print(f"Body oval    : {'on' if args.use_body_oval else 'off'}")
    print(f"n_adjacent   : {args.n_adjacent}")

    train_samples, val_samples = build_samples(args)
    print(f"Total slices : {len(train_samples)} train / {len(val_samples)} val")

    train_ds = DixonThighDataset(
        train_samples, img_size=args.img_size, use_ff=args.use_ff, augment=True,
        n_adjacent=args.n_adjacent, use_body_oval=args.use_body_oval,
    )
    val_ds = DixonThighDataset(
        val_samples, img_size=args.img_size, use_ff=args.use_ff, augment=False,
        n_adjacent=args.n_adjacent, use_body_oval=args.use_body_oval,
    )

    if len(train_ds) == 0:
        raise RuntimeError("No training samples found -- check --gt_base / --sheffield_* / --asian_root paths.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    img_ch = train_ds.n_image_channels
    print(f"Image channels: {img_ch}  ({args.n_adjacent} slices x "
          f"{'2 (water+FF)' if args.use_ff else '1 (water)'})")

    model     = UNet(img_ch=img_ch, base=args.base_ch, t_dim=args.t_dim,
                     dropout=args.dropout).to(device)
    diffusion = GaussianDiffusion(T=args.T, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Body-oval-off checkpoints get a distinct tag so they never collide with
    # the masked variant for the same muscle.
    tag = args.muscle if args.use_body_oval else f"{args.muscle}_nomask"
    ckpt_path = os.path.join(args.out_dir, f"{tag}_latest.pt")
    best_path = os.path.join(args.out_dir, f"{tag}_best.pt")
    start_epoch, best_dice = 0, 0.0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"Resumed from epoch {start_epoch}  (best Dice {best_dice:.4f})")

    log_path = os.path.join(args.out_dir, f"{tag}_log.csv")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,loss,val_dice,lr,elapsed_s\n")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        for img_cpu, mask_cpu in train_loader:
            img_d  = img_cpu.to(device)
            mask_d = mask_cpu.to(device)
            t = torch.randint(0, args.T, (img_d.shape[0],), device=device)
            loss = diffusion.training_loss(model, mask_d, img_d, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0

        val_dice = 0.0
        if len(val_ds) > 0 and (epoch + 1) % args.val_every == 0:
            val_dice = validate(model, diffusion, val_loader, device, args.ddim_steps)
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "best_dice": best_dice, "args": vars(args)}, best_path)
                print(f"  *** Best Dice {best_dice:.4f} -- saved {best_path}")

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:4d}/{args.epochs}  loss={avg_loss:.5f}"
              f"  val_dice={val_dice:.4f}  lr={lr_now:.2e}  t={elapsed:.0f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{val_dice:.4f},{lr_now:.2e},{elapsed:.1f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_dice": best_dice, "args": vars(args),
            }, ckpt_path)

    print(f"\nTraining complete.  Best val Dice: {best_dice:.4f}")
    print(f"Best model : {best_path}")


if __name__ == "__main__":
    main()
