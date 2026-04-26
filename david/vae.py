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
    entropy_decoder_layers: int = 0  # how many first decoder layers get column-entropy reg (0 = off)
    entropy_layer_decay: float = 1.0  # weight decay per layer depth; 1.0 = uniform, 0.5 = halve each layer

    @classmethod
    def from_dict(cls, d: dict) -> "DAVIDConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DAVIDOutput:
    recon: Tensor               # [B, N, D]
    mu: Tensor                  # [B, N, D]
    logvar: Tensor              # [B, N, D]
    m: int                      # prefix length used
    attn_entropy: Tensor | None = None  # scalar: summed column-entropy across all blocks/layers


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


def attention_column_entropy(attn_weights: Tensor) -> Tensor:
    """Weighted column entropy of the attention matrix.

    For each key position j, the column A[:, :, :, j] describes how token j's
    value propagates to all query outputs.  Normalising that column over the query
    dimension gives a probability distribution p_{·j}; its Shannon entropy measures
    how diffuse that influence is.  We weight by j/N so that later tokens are
    penalised more for being broadly influential.

    Args:
        attn_weights: [B, H, N_q, N_k] — row-normalised attention weights (softmax
                      over keys), in any floating-point dtype.

    Returns:
        Scalar tensor: mean weighted column entropy across batch and heads.
    """
    # Work in float32 for numerical stability of log
    w = attn_weights.float()                           # [B, H, N_q, N_k]
    B, H, N_q, N_k = w.shape

    # Column-normalise: p_{ij} = A[i,j] / sum_i A[i,j]
    col_sum = w.sum(dim=2, keepdim=True).clamp(min=1e-8)  # [B, H, 1, N_k]
    p_col = w / col_sum                                     # [B, H, N_q, N_k]

    # Shannon entropy per column: H_j = -sum_i p_{ij} log p_{ij}
    H_j = -(p_col * torch.log(p_col.clamp(min=1e-8))).sum(dim=2)  # [B, H, N_k]

    # Position weights: j/N — later keys penalised more
    j = torch.arange(N_k, device=w.device, dtype=w.dtype)
    weights = j / N_k  # [N_k]

    # Mean over batch, heads, and key positions
    return (H_j * weights).mean()


class SelfAttentionBlock(nn.Module):
    """Self-attention + FFN with pre-norm residuals.

    Uses F.scaled_dot_product_attention (Flash Attention 2 when available) for
    O(N) memory instead of O(N²), which is critical for long video token sequences.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float, ffn_dim: int,
                 progressive_ratio: float = 0.0, compute_attn_entropy: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.progressive_ratio = progressive_ratio
        self.dropout_p = dropout
        self.compute_attn_entropy = compute_attn_entropy
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

    def forward(
        self, x: Tensor, key_padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Returns (output, attn_entropy).

        attn_entropy is a scalar Tensor when compute_attn_entropy=True and
        self.training, otherwise None.  When not None it is part of the
        computation graph and can be included in the training loss.
        """
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

        entropy: Tensor | None = None
        if self.compute_attn_entropy and self.training:
            # Materialise attention weights explicitly (bypasses Flash Attention) so
            # we can compute column entropy.  This is O(N²) memory but we only do it
            # when explicitly requested, and we skip grad_checkpoint on these blocks
            # (see DAVIDEncoder/DAVIDDecoder) since we're already storing the matrix.
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]
            if attn_mask is not None:
                scores = scores + attn_mask
            attn_weights = F.softmax(scores, dim=-1)  # [B, H, N_q, N_k]
            if self.dropout_p > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.dropout_p)
            h = torch.matmul(attn_weights, v)          # [B, H, N, head_dim]
            entropy = attention_column_entropy(attn_weights)
        else:
            h = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        h = h.transpose(1, 2).reshape(B, N, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, entropy


class DAVIDEncoder(nn.Module):
    """Self-attention with progressive masking: early tokens see all, late tokens see less."""

    def __init__(self, config: DAVIDConfig):
        super().__init__()
        D = config.input_dim
        ffn_dim = D * config.ffn_multiplier
        self.grad_checkpoint = config.grad_checkpoint
        # Encoder processes temporal frames — no entropy reg (all blocks use Flash Attention)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(D, config.n_heads, config.dropout, ffn_dim,
                               progressive_ratio=config.progressive_ratio)
            for _ in range(config.n_encoder_layers)
        ])
        self.out_proj = nn.Linear(D, 2 * D)

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """
        Args:
            features: [B, N, D]
            mask:     [B, N] True = valid
        Returns:
            z, mu, logvar: each [B, N, D]
            total_entropy: always None (encoder has no entropy regularisation)
        """
        key_padding_mask = ~mask
        x = features
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x, _ = grad_ckpt(block, x, key_padding_mask, use_reentrant=False)
            else:
                x, _ = block(x, key_padding_mask=key_padding_mask)

        out = self.out_proj(x)  # [B, N, 2*D]
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 4.0)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar, None  # encoder never produces entropy


class DAVIDDecoder(nn.Module):
    """Self-attention with progressive masking: reconstructs N tokens from zero-padded prefix."""

    def __init__(self, config: DAVIDConfig):
        super().__init__()
        D = config.input_dim
        ffn_dim = D * config.ffn_multiplier
        self.grad_checkpoint = config.grad_checkpoint
        # Only the first entropy_decoder_layers blocks compute column-entropy
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(D, config.n_heads, config.dropout, ffn_dim,
                               progressive_ratio=config.progressive_ratio,
                               compute_attn_entropy=(i < config.entropy_decoder_layers))
            for i in range(config.n_decoder_layers)
        ])
        self.entropy_layer_decay = config.entropy_layer_decay
        self.out_proj = nn.Linear(D, D)

    def forward(self, z_padded: Tensor) -> tuple[Tensor, Tensor | None]:
        """Returns (reconstructed features, total_entropy or None).

        total_entropy is a decay-weighted sum of column-entropy from the first
        entropy_decoder_layers blocks.  Layer i contributes entropy_layer_decay^i
        times its entropy, so layer 0 (closest to DAVID tokens) has the highest
        weight and deeper layers contribute progressively less.  Later blocks use
        Flash Attention and contribute no entropy term.
        """
        x = z_padded
        total_entropy: Tensor | None = None
        for i, block in enumerate(self.blocks):
            # Use grad_checkpoint only on blocks that don't materialise the attention matrix
            if self.grad_checkpoint and self.training and not block.compute_attn_entropy:
                x, entropy = grad_ckpt(block, x, None, use_reentrant=False)
            else:
                x, entropy = block(x)
            if entropy is not None:
                w = self.entropy_layer_decay ** i
                weighted = w * entropy
                total_entropy = weighted if total_entropy is None else total_entropy + weighted
        return self.out_proj(x), total_entropy


class DAVIDVAE(nn.Module):
    """Encoder → sample z → prefix truncate → zero-pad → decoder."""

    def __init__(self, config: DAVIDConfig):
        super().__init__()
        self.config = config
        self.encoder = DAVIDEncoder(config)
        self.decoder = DAVIDDecoder(config)

    def forward(self, features: Tensor, mask: Tensor, training: bool = True,
                m: int | None = None) -> DAVIDOutput:
        z, mu, logvar, enc_entropy = self.encoder(features, mask)

        N = z.shape[1]
        if m is None:
            m = torch.randint(1, N + 1, (1,)).item() if training else N

        z_padded = torch.zeros_like(z)
        z_padded[:, :m, :] = z[:, :m, :]

        recon, dec_entropy = self.decoder(z_padded)

        # Sum encoder and decoder entropies; either may be None when disabled
        attn_entropy: Tensor | None = None
        for e in (enc_entropy, dec_entropy):
            if e is not None:
                attn_entropy = e if attn_entropy is None else attn_entropy + e

        return DAVIDOutput(recon=recon, mu=mu, logvar=logvar, m=m, attn_entropy=attn_entropy)

    def encode(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z, mu, logvar, _ = self.encoder(features, mask)
        return z, mu, logvar

    def decode(self, z_prefix: Tensor, n: int) -> Tensor:
        z_padded = z_prefix.new_zeros(z_prefix.shape[0], n, z_prefix.shape[2])
        z_padded[:, :z_prefix.shape[1], :] = z_prefix
        recon, _ = self.decoder(z_padded)
        return recon
