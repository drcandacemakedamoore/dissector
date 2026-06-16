r"""Training script for onzediff — 2.5D diffusion segmentation with body-oval masking.

Usage
-----
python train.py \
    --muscle R_gracilis \
    --gt_base ~/myosegmenTUM \
    --out_dir ~/onzediff_ckpts \
    --epochs 300 \
    --batch_size 8 \
    --n_adjacent 3 \
    --use_ff
"""

from __future__ import annotations
import argparse
import os
import time
import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset import GT_LABELS, DixonThighDataset, discover_subjects, train_val_split
from diffusion import GaussianDiffusion
from unet import UNet


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train onzediff on myosegmenTUM")
    p.add_argument("--muscle",      required=True, choices=list(GT_LABELS))
    p.add_argument("--gt_base",     default=os.path.expanduser("~/myosegmenTUM"))
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
                   help="Include fat-fraction as extra channel for each adjacent slice")
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
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = diffusion.ddim_sample(model, img, num_steps=ddim_steps)
        scores.append(diffusion.dice((pred > 0).float(), (mask > 0).float()))
    model.train()
    return float(sum(scores) / max(len(scores), 1))


def main() -> None:
    """Train a binary onzediff model for one muscle."""
    args = get_args()
    if args.n_adjacent % 2 == 0:
        raise ValueError("--n_adjacent must be odd (e.g. 1, 3, 5)")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Muscle     : {args.muscle}")
    print(f"n_adjacent : {args.n_adjacent}")

    all_subjects = discover_subjects(args.gt_base)
    train_subj, val_subj = train_val_split(all_subjects, val_fraction=args.val_frac)
    print(f"Subjects   : {len(train_subj)} train / {len(val_subj)} val")

    train_ds = DixonThighDataset(
        args.gt_base, args.muscle, train_subj,
        img_size=args.img_size, use_ff=args.use_ff, augment=True,
        n_adjacent=args.n_adjacent,
    )
    val_ds = DixonThighDataset(
        args.gt_base, args.muscle, val_subj,
        img_size=args.img_size, use_ff=args.use_ff, augment=False,
        n_adjacent=args.n_adjacent,
    )
    print(f"Slices     : {len(train_ds)} train / {len(val_ds)} val")

    if len(train_ds) == 0:
        raise RuntimeError("No training samples found — check --gt_base path.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    img_ch = train_ds.n_image_channels
    print(f"Image channels: {img_ch}  ({args.n_adjacent} slices × "
          f"{'2 (water+FF)' if args.use_ff else '1 (water)'})")

    model     = UNet(img_ch=img_ch, base=args.base_ch, t_dim=args.t_dim,
                     dropout=args.dropout).to(device)
    diffusion = GaussianDiffusion(T=args.T, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    ckpt_path = os.path.join(args.out_dir, f"{args.muscle}_latest.pt")
    best_path = os.path.join(args.out_dir, f"{args.muscle}_best.pt")
    start_epoch, best_dice = 0, 0.0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"Resumed from epoch {start_epoch}  (best Dice {best_dice:.4f})")

    log_path = os.path.join(args.out_dir, f"{args.muscle}_log.csv")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,loss,val_dice,lr,elapsed_s\n")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            t = torch.randint(0, args.T, (img.shape[0],), device=device)
            loss = diffusion.training_loss(model, mask, img, t)
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
                print(f"  *** Best Dice {best_dice:.4f} — saved {best_path}")

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
