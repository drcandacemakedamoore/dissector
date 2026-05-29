"""
4-level U-Net for diffusion segmentation.

Architecture
------------
Input:  (B, img_ch + 1, H, W)  — conditioning image(s) concatenated with noisy mask
Output: (B, 1, H, W)            — predicted noise on the mask channel

Timestep conditioning is injected into every residual block via AdaGN
(scale + shift derived from the timestep MLP).  Self-attention is applied
at the two deepest encoder / decoder levels.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────

def _groups(ch: int) -> int:
    """Largest power-of-2 divisor of ch, capped at 32."""
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class SinusoidalPE(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device).float()
            / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([args.cos(), args.sin()], dim=1)     # (B, dim)


class TimestepMLP(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.pe = SinusoidalPE(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.pe(t))


# ── building blocks ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with AdaGN timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1  = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch * 2)   # → scale + shift
        self.drop   = nn.Dropout(dropout)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        ts = self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)   # (B, 2*out_ch, 1, 1)
        scale, shift = ts.chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """Single-head spatial self-attention (efficient einsum implementation)."""

    def __init__(self, ch: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.norm  = nn.GroupNorm(_groups(ch), ch)
        self.qkv   = nn.Conv2d(ch, ch * 3, 1)
        self.proj  = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(dim=1)                             # each (B, heads, ch/heads, HW)
        scale = (C // self.heads) ** -0.5
        attn = torch.einsum('bhdi,bhdj->bhij', q * scale, k).softmax(dim=-1)
        h = torch.einsum('bhij,bhdj->bhdi', attn, v).reshape(B, C, H, W)
        return x + self.proj(h)


# ── U-Net ─────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Parameters
    ----------
    img_ch  : conditioning image channels (1 = water only, 2 = water + FF)
    base    : base channel count (default 64)
    t_dim   : sinusoidal embedding dimension (default 256; MLP output = t_dim*4)
    dropout : dropout inside residual blocks
    """

    def __init__(
        self,
        img_ch: int = 1,
        base: int = 64,
        t_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        td = t_dim * 4     # timestep MLP output dimension
        ic = img_ch + 1    # input channels: image(s) + noisy mask

        self.t_emb = TimestepMLP(t_dim, td)

        # ── Encoder ──────────────────────────────────────────────────────────
        # e0: (ic → base),     256×256
        # e1: (base → base*2), 128×128
        # e2: (base*2 → base*4), 64×64  + attention
        # e3: (base*4 → base*8), 32×32  + attention
        # bottleneck at 16×16
        self.enc_init = nn.Conv2d(ic, base, 3, padding=1)
        self.enc1a = ResBlock(base,     base * 2, td, dropout)
        self.enc1b = ResBlock(base * 2, base * 2, td, dropout)
        self.enc2a = ResBlock(base * 2, base * 4, td, dropout)
        self.enc2b = ResBlock(base * 4, base * 4, td, dropout)
        self.attn2 = Attention(base * 4)
        self.enc3a = ResBlock(base * 4, base * 8, td, dropout)
        self.enc3b = ResBlock(base * 8, base * 8, td, dropout)
        self.attn3 = Attention(base * 8)
        self.down  = nn.AvgPool2d(2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid1      = ResBlock(base * 8, base * 8, td, dropout)
        self.mid_attn  = Attention(base * 8)
        self.mid2      = ResBlock(base * 8, base * 8, td, dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        # skip channels concatenated: base*8, base*4, base*2, base
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.dec3a  = ResBlock(base * 8 + base * 8, base * 4, td, dropout)
        self.dec3b  = ResBlock(base * 4,             base * 4, td, dropout)
        self.dattn3 = Attention(base * 4)

        self.dec2a  = ResBlock(base * 4 + base * 4, base * 2, td, dropout)
        self.dec2b  = ResBlock(base * 2,             base * 2, td, dropout)
        self.dattn2 = Attention(base * 2)

        self.dec1a  = ResBlock(base * 2 + base * 2, base, td, dropout)
        self.dec1b  = ResBlock(base,                 base, td, dropout)

        self.dec0   = ResBlock(base + base, base, td, dropout)

        self.out = nn.Sequential(
            nn.GroupNorm(_groups(base), base),
            nn.SiLU(),
            nn.Conv2d(base, 1, 1),
        )

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, img_ch+1, H, W)  — concat(img, x_t)
        t : (B,)                  — integer diffusion timesteps
        Returns (B, 1, H, W) predicted noise.
        """
        te = self.t_emb(t)   # (B, t_dim*4)

        # Encoder
        s0 = self.enc_init(x)                    # (B, base,   H,   W)

        h = self.down(s0)
        h = self.enc1a(h, te)
        s1 = self.enc1b(h, te)                   # (B, base*2, H/2, W/2)

        h = self.down(s1)
        h = self.enc2a(h, te)
        h = self.enc2b(h, te)
        s2 = self.attn2(h)                       # (B, base*4, H/4, W/4)

        h = self.down(s2)
        h = self.enc3a(h, te)
        h = self.enc3b(h, te)
        s3 = self.attn3(h)                       # (B, base*8, H/8, W/8)

        # Bottleneck
        h = self.down(s3)                        # (B, base*8, H/16, W/16)
        h = self.mid1(h, te)
        h = self.mid_attn(h)
        h = self.mid2(h, te)

        # Decoder
        h = self.up(h)
        h = self.dec3a(torch.cat([h, s3], dim=1), te)
        h = self.dec3b(h, te)
        h = self.dattn3(h)

        h = self.up(h)
        h = self.dec2a(torch.cat([h, s2], dim=1), te)
        h = self.dec2b(h, te)
        h = self.dattn2(h)

        h = self.up(h)
        h = self.dec1a(torch.cat([h, s1], dim=1), te)
        h = self.dec1b(h, te)

        h = self.up(h)
        h = self.dec0(torch.cat([h, s0], dim=1), te)

        return self.out(h)
