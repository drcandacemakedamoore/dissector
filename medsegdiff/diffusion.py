"""
DDPM / DDIM diffusion utilities for binary mask segmentation.

Forward process  : q(x_t | x_0) = N(sqrt(ᾱ_t)·x_0, (1-ᾱ_t)·I)
Training loss    : MSE(ε_θ(x_t, t, img), ε)
Sampling         : DDIM deterministic reverse (eta=0) or stochastic (eta>0)

Masks live in [-1, +1].  After sampling, threshold at 0 to recover {0,1}.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02
                           ) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


class GaussianDiffusion:
    """
    Parameters
    ----------
    T           : total diffusion steps (1 000 is standard)
    beta_start  : noise schedule start value
    beta_end    : noise schedule end value
    device      : torch device
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = 'cpu',
    ) -> None:
        self.T = T
        betas = _linear_beta_schedule(T, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register = lambda name, val: setattr(self, name, val)
        self.betas       = betas
        self.alphas      = alphas
        self.alpha_bars  = alpha_bars                        # ᾱ_t
        self.sqrt_ab     = alpha_bars.sqrt()                 # √ᾱ_t
        self.sqrt_1m_ab  = (1.0 - alpha_bars).sqrt()        # √(1-ᾱ_t)

    # ── forward process ───────────────────────────────────────────────────────

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t ~ q(x_t | x_0) using the closed-form marginal.

        Returns (x_t, noise) — noise is generated here if not supplied.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        ab  = self.sqrt_ab[t].view(-1, 1, 1, 1)
        s1m = self.sqrt_1m_ab[t].view(-1, 1, 1, 1)
        return ab * x0 + s1m * noise, noise

    # ── training loss ─────────────────────────────────────────────────────────

    def training_loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        img: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard DDPM ε-prediction loss.

        x0  : (B, 1, H, W) clean mask in [-1, 1]
        img : (B, C, H, W) conditioning image in [-1, 1]
        t   : (B,) integer timesteps sampled uniformly from [0, T)
        """
        xt, noise = self.q_sample(x0, t)
        model_in  = torch.cat([img, xt], dim=1)          # (B, C+1, H, W)
        pred      = model(model_in, t)
        return F.mse_loss(pred, noise)

    # ── DDIM sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        img: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM reverse diffusion conditioned on img.

        img       : (B, C, H, W) conditioning image in [-1, 1]
        num_steps : number of denoising steps (default 50)
        eta       : stochasticity; 0 = deterministic DDIM, 1 = DDPM

        Returns (B, 1, H, W) predicted mask in [-1, 1].
        Threshold at 0 to get a binary {0, 1} mask.
        """
        B, _, H, W = img.shape
        device = img.device

        # Evenly spaced timestep schedule, decreasing
        steps = torch.linspace(self.T - 1, 0, num_steps, dtype=torch.long, device=device)

        xt = torch.randn(B, 1, H, W, device=device)

        for i, t_val in enumerate(steps):
            t = torch.full((B,), int(t_val), device=device, dtype=torch.long)

            model_in   = torch.cat([img, xt], dim=1)
            pred_noise = model(model_in, t)

            ab_t     = self.alpha_bars[t_val]
            ab_prev  = self.alpha_bars[steps[i + 1]] if i + 1 < num_steps else torch.tensor(1.0, device=device)

            # Predicted x_0 from current estimate
            x0_pred = (xt - (1.0 - ab_t).sqrt() * pred_noise) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM update
            sigma   = eta * ((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)).sqrt()
            noise   = torch.randn_like(xt) if eta > 0 else torch.zeros_like(xt)
            xt      = ab_prev.sqrt() * x0_pred \
                    + (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * pred_noise \
                    + sigma * noise

        return xt   # in [-1, 1]; threshold at 0 for binary mask

    # ── evaluation helpers ────────────────────────────────────────────────────

    @staticmethod
    def dice(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6) -> float:
        """Dice coefficient between two boolean / {0,1} tensors."""
        inter = (pred_bin * gt_bin).sum().float()
        total = pred_bin.sum().float() + gt_bin.sum().float()
        return float(2.0 * inter / (total + eps))
