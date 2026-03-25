"""Loss functions and beta scheduler for DAVID VAE training."""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass


def reconstruction_loss(recon: Tensor, target: Tensor) -> Tensor:
    """Mean squared error between reconstructed and target features.

    Args:
        recon:  [batch, N_queries, D] — decoder output.
        target: [batch, N_queries, D] — interpolated backbone features.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(recon, target)


def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL divergence from N(mu, sigma^2) to N(0, 1), averaged over all dims.

    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Args:
        mu:     [batch, L, D]
        logvar: [batch, L, D]

    Returns:
        Scalar KL loss (mean over batch, L, D).
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
    """Combined DAVID VAE loss: MSE + beta * KL.

    Args:
        recon:  [B, N_queries, D]
        target: [B, N_queries, D]
        mu:     [B, L, D]
        logvar: [B, L, D]
        beta:   KL weight (annealed from 0 during training)
        m:      Truncation index (for logging)

    Returns:
        LossOutput with total loss tensor and scalar metrics.
    """
    mse = reconstruction_loss(recon, target)
    kl = kl_loss(mu, logvar)
    total = mse + beta * kl
    return LossOutput(total=total, mse=mse.item(), kl=kl.item(), beta=beta, m=m)


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
