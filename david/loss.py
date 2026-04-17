"""Loss functions and schedulers for DAVID VAE training."""

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
    m: int              # truncation index used
    entropy: float = 0.0  # attention column-entropy term (0 when disabled)


def david_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float,
    m: int,
    lambda_entropy: float = 0.0,
    attn_entropy: Tensor | None = None,
) -> LossOutput:
    """Combined DAVID VAE loss: adaptive recon + beta * KL + lambda * attn_entropy.

    Args:
        recon:          [B, N, D]
        target:         [B, N, D]
        mu:             [B, N, D]
        logvar:         [B, N, D]
        beta:           KL weight (annealed from 0)
        m:              Truncation prefix length
        lambda_entropy: Weight for attention column-entropy regularisation (annealed from 0)
        attn_entropy:   Scalar Tensor from DAVIDVAE (None when compute_attn_entropy=False)

    Returns:
        LossOutput with total loss tensor and scalar metrics.
    """
    N = recon.shape[1]
    recon_loss = reconstruction_loss(recon, target, m, N)
    kl = kl_loss(mu, logvar)
    total = recon_loss + beta * kl

    entropy_val = 0.0
    if attn_entropy is not None and lambda_entropy > 0.0:
        total = total + lambda_entropy * attn_entropy
        entropy_val = attn_entropy.item()

    return LossOutput(
        total=total,
        mse=recon_loss.item(),
        kl=kl.item(),
        beta=beta,
        m=m,
        entropy=entropy_val,
    )


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


class LambdaScheduler:
    """Linear warm-up scheduler for the attention entropy regularisation weight λ.

    lambda = 0 for steps 0..warmup_start
    lambda linearly ramps from 0 to lambda_target over warmup_start..warmup_end
    lambda = lambda_target for steps > warmup_end

    Mirrors BetaScheduler exactly; kept separate so the two weights can be tuned
    independently and logged with distinct names.
    """

    def __init__(self, lambda_target: float, warmup_start: int, warmup_end: int):
        self.lambda_target = lambda_target
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end

    def get_lambda(self, step: int) -> float:
        if step <= self.warmup_start:
            return 0.0
        if step >= self.warmup_end:
            return self.lambda_target
        progress = (step - self.warmup_start) / (self.warmup_end - self.warmup_start)
        return self.lambda_target * progress
