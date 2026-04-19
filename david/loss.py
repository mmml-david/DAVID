"""Loss functions and beta scheduler for DAVID VAE training."""

import math
import torch
from torch import Tensor
from dataclasses import dataclass


def reconstruction_loss(recon: Tensor, target: Tensor, m: int, N: int) -> Tensor:
    """Adaptive reconstruction loss that scales with prefix length m.

    Exponent p = 2 * sigmoid(m/N) smoothly transitions from ~1 (L1) to ~2 (L2).
    Scale s = 2 * sigmoid(m/N) weights the loss: low when m is small, high when m is large.

    Args:
        recon:  [batch, N, D] — decoder output.
        target: [batch, N, D] — original backbone features.
        m:      prefix length used in this forward pass.
        N:      full sequence length.

    Returns:
        Scalar reconstruction loss.
    """
    t = torch.sigmoid(torch.tensor(float(m) / N, device=recon.device))
    p = 2.0 * t      # exponent: ~1 → ~2
    scale = 2.0 * t   # weight:   ~1 → ~2

    return scale * (recon - target).abs().pow(p).mean()


def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL divergence from N(mu, sigma^2) to N(0, 1), averaged over all dims.

    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Args:
        mu:     [batch, N, D]
        logvar: [batch, N, D]

    Returns:
        Scalar KL loss (mean over batch, N, D).
    """
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()


@dataclass
class LossOutput:
    total: Tensor
    mse: float
    kl: float
    beta: float
    m: int  # truncation index used


def david_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float,
    m: int,
) -> LossOutput:
    """Combined DAVID VAE loss: adaptive recon + beta * KL.

    Args:
        recon:  [B, N, D]
        target: [B, N, D]
        mu:     [B, N, D]
        logvar: [B, N, D]
        beta:   KL weight (keep small, annealed from 0 during training)
        m:      Truncation prefix length

    Returns:
        LossOutput with total loss tensor and scalar metrics.
    """
    N = recon.shape[1]
    recon_loss = reconstruction_loss(recon, target, m, N)
    kl = kl_loss(mu, logvar)
    total = recon_loss + beta * kl
    return LossOutput(total=total, mse=recon_loss.item(), kl=kl.item(), beta=beta, m=m)


class BetaScheduler:
    """Linear beta warm-up scheduler.

    beta = 0 for steps 0..warmup_start
    beta linearly ramps from 0 to beta_target over warmup_start..warmup_end
    beta = beta_target for steps > warmup_end
    """

    def __init__(self, beta_target: float, warmup_start: int, warmup_end: int):
        self.beta_target = beta_target
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end

    def get_beta(self, step: int) -> float:
        if step <= self.warmup_start:
            return 0.0
        if step >= self.warmup_end:
            return self.beta_target
        progress = (step - self.warmup_start) / (self.warmup_end - self.warmup_start)
        return self.beta_target * progress


class MRatioScheduler:
    """Linear schedule for sliding prefix-ratio sampling window.

    m is sampled from a ratio window [lo_ratio, hi_ratio] over N tokens:
      m ~ Uniform(floor(N * lo_ratio), floor(N * hi_ratio))

    The window endpoints are linearly scheduled:
      lo_ratio: lo_start -> lo_end
      hi_ratio: hi_start -> hi_end

    Ratios are not clipped to 1.0 so values like hi_ratio=1.1 are allowed, and
    converted token bounds are clamped into [0, N].
    """

    def __init__(
        self,
        lo_start: float,
        hi_start: float,
        lo_end: float,
        hi_end: float,
        warmup_start: int,
        warmup_end: int,
    ):
        self.lo_start = float(lo_start)
        self.hi_start = float(hi_start)
        self.lo_end = float(lo_end)
        self.hi_end = float(hi_end)
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end

    @staticmethod
    def _lerp(start: float, end: float, progress: float) -> float:
        return start + progress * (end - start)

    def get_ratio_window(self, step: int) -> tuple[float, float]:
        if step <= self.warmup_start:
            lo, hi = self.lo_start, self.hi_start
        elif step >= self.warmup_end:
            lo, hi = self.lo_end, self.hi_end
        else:
            progress = (step - self.warmup_start) / (self.warmup_end - self.warmup_start)
            lo = self._lerp(self.lo_start, self.lo_end, progress)
            hi = self._lerp(self.hi_start, self.hi_end, progress)
        return (min(lo, hi), max(lo, hi))

    def sample_m(self, step: int, n_tokens: int) -> int:
        lo_ratio, hi_ratio = self.get_ratio_window(step)
        lower = max(0, min(n_tokens, math.floor(n_tokens * lo_ratio)))
        upper = max(0, min(n_tokens, math.floor(n_tokens * hi_ratio)))
        if upper < lower:
            lower, upper = upper, lower
        return torch.randint(lower, upper + 1, (1,)).item()
