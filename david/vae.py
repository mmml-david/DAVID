"""DAVID VAE: self-attention encoder/decoder with stochastic prefix truncation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_ckpt
from dataclasses import dataclass


@dataclass
class DAVIDConfig:
    input_dim: int = 4096       # Qwen3-VL pooler_output dim (= latent dim)
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_heads: int = 16
    dropout: float = 0.1
    ffn_multiplier: int = 4
    progressive_ratio: float = 0.0  # 0.0 = no mask, 1.0 = token N-1 sees ~1 random token
    grad_checkpoint: bool = False   # recompute activations during backward to save memory

    @classmethod
    def from_dict(cls, d: dict) -> "DAVIDConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DAVIDOutput:
    recon: Tensor     # [B, N, D]
    mu: Tensor        # [B, N, D]
    logvar: Tensor    # [B, N, D]
    m: int            # prefix length used


def progressive_attn_mask(n: int, ratio: float, device) -> Tensor | None:
    """Stochastic progressive attention mask (training only).

    Token i keeps ceil(N - i * ratio * (N-1)/N) random attention targets.
      ratio=0.0: no masking
      ratio=0.5: token N-1 sees ~N/2 random tokens
      ratio=1.0: token N-1 sees ~1 random token

    Returns [N, N] bool mask where True = ignore (PyTorch MHA convention), or None.
    """
    if ratio <= 0.0:
        return None
    idx = torch.arange(n, device=device, dtype=torch.float32)
    # Number of tokens to DROP for each row
    n_drop = (idx * ratio * (n - 1) / n).floor().long()  # [N], token 0 drops 0
    # For each row, generate random scores and drop the lowest-scored ones
    scores = torch.rand(n, n, device=device)  # [N, N]
    # Rank positions per row (argsort of argsort = rank)
    ranks = scores.argsort(dim=1).argsort(dim=1)  # [N, N], 0 = lowest score
    # Mask positions with rank < n_drop[i]
    return ranks < n_drop.unsqueeze(1)  # [N, N], True = ignore


class SelfAttentionBlock(nn.Module):
    """Self-attention + FFN with pre-norm residuals.

    Uses F.scaled_dot_product_attention (Flash Attention 2 when available) for
    O(N) memory instead of O(N²), which is critical for long video token sequences.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float, ffn_dim: int,
                 progressive_ratio: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.progressive_ratio = progressive_ratio
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        B, N, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Build additive float mask: 0 = attend, -inf = ignore
        attn_mask = None
        if self.training and self.progressive_ratio > 0.0:
            prog = progressive_attn_mask(N, self.progressive_ratio, x.device)  # [N,N] bool
            if prog is not None:
                attn_mask = torch.zeros(1, 1, N, N, device=x.device, dtype=x.dtype)
                attn_mask.masked_fill_(prog.unsqueeze(0).unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: [B, N], True = padding token
            pad = torch.zeros(B, 1, 1, N, device=x.device, dtype=x.dtype)
            pad.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))
            attn_mask = pad if attn_mask is None else attn_mask + pad

        h = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        h = h.transpose(1, 2).reshape(B, N, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class DAVIDEncoder(nn.Module):
    """Self-attention with progressive masking: early tokens see all, late tokens see less."""

    def __init__(self, config: DAVIDConfig):
        super().__init__()
        D = config.input_dim
        ffn_dim = D * config.ffn_multiplier
        self.grad_checkpoint = config.grad_checkpoint
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(D, config.n_heads, config.dropout, ffn_dim,
                               progressive_ratio=config.progressive_ratio)
            for _ in range(config.n_encoder_layers)
        ])
        self.out_proj = nn.Linear(D, 2 * D)

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            features: [B, N, D]
            mask:     [B, N] True = valid
        Returns:
            z, mu, logvar: each [B, N, D]
        """
        key_padding_mask = ~mask
        x = features
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = grad_ckpt(block, x, key_padding_mask, use_reentrant=False)
            else:
                x = block(x, key_padding_mask=key_padding_mask)

        out = self.out_proj(x)  # [B, N, 2*D]
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 4.0)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar


class DAVIDDecoder(nn.Module):
    """Self-attention with progressive masking: reconstructs N tokens from zero-padded prefix."""

    def __init__(self, config: DAVIDConfig):
        super().__init__()
        D = config.input_dim
        ffn_dim = D * config.ffn_multiplier
        self.grad_checkpoint = config.grad_checkpoint
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(D, config.n_heads, config.dropout, ffn_dim,
                               progressive_ratio=config.progressive_ratio)
            for _ in range(config.n_decoder_layers)
        ])
        self.out_proj = nn.Linear(D, D)

    def forward(self, z_padded: Tensor) -> Tensor:
        x = z_padded
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.out_proj(x)


class DAVIDVAE(nn.Module):
    """Encoder → sample z → prefix truncate → zero-pad → decoder."""

    def __init__(self, config: DAVIDConfig):
        super().__init__()
        self.config = config
        self.encoder = DAVIDEncoder(config)
        self.decoder = DAVIDDecoder(config)

    def forward(self, features: Tensor, mask: Tensor, training: bool = True,
                m: int | None = None) -> DAVIDOutput:
        z, mu, logvar = self.encoder(features, mask)

        N = z.shape[1]
        if m is None:
            m = torch.randint(1, N + 1, (1,)).item() if training else N

        z_padded = torch.zeros_like(z)
        z_padded[:, :m, :] = z[:, :m, :]

        recon = self.decoder(z_padded)
        return DAVIDOutput(recon=recon, mu=mu, logvar=logvar, m=m)

    def encode(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.encoder(features, mask)

    def decode(self, z_prefix: Tensor, n: int) -> Tensor:
        z_padded = z_prefix.new_zeros(z_prefix.shape[0], n, z_prefix.shape[2])
        z_padded[:, :z_prefix.shape[1], :] = z_prefix
        return self.decoder(z_padded)
