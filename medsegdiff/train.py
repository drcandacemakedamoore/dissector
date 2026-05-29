r"""Training script for diffusion-based thigh muscle segmentation.

Usage
-----
python train.py \\
    --muscle R_gracilis \\
    --gt_base ~/myosegmenTUM \\
    --out_dir ~/medsegdiff_ckpts \\
    --epochs 300 \\
    --batch_size 8 \\
    --img_size 256 \\
    --base_ch 64 \\
    --use_ff

One checkpoint is saved per muscle.  Training is resumable: if a checkpoint
already exists the script loads it and continues from the saved epoch.
"""

from __future__ import annotations
import argparse
import os
import time
import torch
from dataset import GT_LABELS
from dataset import DixonThighDataset
from dataset import discover_subjects
from dataset import train_val_split
from diffusion import GaussianDiffusion
from torch import optim
from torch.utils.data import DataLoader
from unet import UNet

# ── argument parsing ──────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    p = argparse.ArgumentParser(description="Train MedSegDiff on myosegmenTUM")
    p.add_argument("--muscle",     required=True, choices=list(GT_LABELS),
                   help="Which muscle to segment")
    p.add_argument("--gt_base",    default=os.path.expanduser("~/myosegmenTUM"),
                   help="Path to myosegmenTUM root directory")
    p.add_argument("--out_dir",    default=os.path.expanduser("~/medsegdiff_ckpts"),
                   help="Directory for checkpoints and logs")
    p.add_argument("--epochs",     type=int,   default=300)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--img_size",   type=int,   default=256)
    p.add_argument("--base_ch",    type=int,   default=64,
                   help="U-Net base channel count")
    p.add_argument("--t_dim",      type=int,   default=256,
                   help="Timestep embedding dimension")
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--T",          type=int,   default=1000,
                   help="Diffusion steps")
    p.add_argument("--use_ff",     action="store_true",
                   help="Use fat-fraction as second image channel")
    p.add_argument("--val_frac",   type=float, default=0.2)
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--save_every", type=int,   default=25,
                   help="Save checkpoint every N epochs")
    p.add_argument("--ddim_steps", type=int,   default=50,
                   help="DDIM steps for validation Dice")
    p.add_argument("--val_every",  type=int,   default=25,
                   help="Run validation every N epochs (expensive)")
    return p.parse_args()


# ── validation ────────────────────────────────────────────────────────────────

def validate(model, diffusion, loader, device, ddim_steps):
    """Run DDIM inference on validation loader and return mean Dice coefficient."""
    model.eval()
    dice_scores = []
    for raw_img, raw_mask in loader:
        batch_img = raw_img.to(device)
        batch_mask = raw_mask.to(device)
        pred = diffusion.ddim_sample(model, batch_img, num_steps=ddim_steps)
        pred_bin = (pred > 0.0).float()
        gt_bin   = (batch_mask > 0.0).float()
        dice_scores.append(diffusion.dice(pred_bin, gt_bin))
    model.train()
    return float(sum(dice_scores) / max(len(dice_scores), 1))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Train a binary diffusion segmentation model for a single muscle."""
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Muscle : {args.muscle}")

    # ── datasets ─────────────────────────────────────────────────────────────
    all_subjects = discover_subjects(args.gt_base)
    train_subj, val_subj = train_val_split(all_subjects, val_fraction=args.val_frac)
    print(f"Subjects: {len(train_subj)} train / {len(val_subj)} val")

    train_ds = DixonThighDataset(
        args.gt_base, args.muscle, train_subj,
        img_size=args.img_size, use_ff=args.use_ff, augment=True,
    )
    val_ds = DixonThighDataset(
        args.gt_base, args.muscle, val_subj,
        img_size=args.img_size, use_ff=args.use_ff, augment=False,
    )
    print(f"Slices : {len(train_ds)} train / {len(val_ds)} val")

    if len(train_ds) == 0:
        raise RuntimeError("No training samples found — check --gt_base path.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    img_ch = train_ds.n_image_channels
    print(f"Image channels: {img_ch}")

    # ── model & diffusion ─────────────────────────────────────────────────────
    model = UNet(img_ch=img_ch, base=args.base_ch, t_dim=args.t_dim,
                 dropout=args.dropout).to(device)
    diffusion = GaussianDiffusion(T=args.T, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── resume if checkpoint exists ───────────────────────────────────────────
    ckpt_path = os.path.join(args.out_dir, f"{args.muscle}_latest.pt")
    best_path = os.path.join(args.out_dir, f"{args.muscle}_best.pt")
    start_epoch = 0
    best_dice = 0.0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"Resumed from epoch {start_epoch}  (best Dice so far: {best_dice:.4f})")

    # ── training loop ─────────────────────────────────────────────────────────
    log_path = os.path.join(args.out_dir, f"{args.muscle}_log.csv")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,loss,val_dice,lr,elapsed_s\n")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        for raw_img, raw_mask in train_loader:
            batch_img = raw_img.to(device)
            batch_mask = raw_mask.to(device)
            batch_size = batch_img.shape[0]
            t = torch.randint(0, args.T, (batch_size,), device=device)

            loss = diffusion.training_loss(model, batch_mask, batch_img, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0

        # Validation
        val_dice = 0.0
        if len(val_ds) > 0 and (epoch + 1) % args.val_every == 0:
            val_dice = validate(model, diffusion, val_loader, device, args.ddim_steps)
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "best_dice": best_dice, "args": vars(args)}, best_path)
                print(f"  *** New best Dice {best_dice:.4f} — saved {best_path}")

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:4d}/{args.epochs}  loss={avg_loss:.5f}"
              f"  val_dice={val_dice:.4f}  lr={lr_now:.2e}  t={elapsed:.0f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{val_dice:.4f},{lr_now:.2e},{elapsed:.1f}\n")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save({
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch":     epoch,
                "best_dice": best_dice,
                "args":      vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint -> {ckpt_path}")

    print(f"\nTraining complete.  Best val Dice: {best_dice:.4f}")
    print(f"Best model : {best_path}")


if __name__ == "__main__":
    main()
