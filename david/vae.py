"""DAVID VAE: encoder, decoder, and full model with stochastic prefix truncation."""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field


@dataclass
class DAVIDConfig:
    input_dim: int = 2048       # Qwen3-VL pooler_output dim
    latent_dim: int = 2048      # latent space dim (same as input for easy init)
    L: int = 64                 # number of DAVID latent tokens
    N_queries: int = 256        # fixed number of decoder output tokens
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_heads: int = 16
    dropout: float = 0.1
    ffn_multiplier: int = 4     # FFN hidden dim = latent_dim * ffn_multiplier

    @classmethod
    def from_dict(cls, d: dict) -> "DAVIDConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DAVIDOutput:
    recon: Tensor     # [B, N_queries, D] reconstructed features
    mu: Tensor        # [B, L, D]
    logvar: Tensor    # [B, L, D]
    m: int            # truncation index used during this forward pass


class CrossAttentionBlock(nn.Module):
    """Single block: cross-attention (queries → context) + self-attention + FFN."""

    def __init__(self, dim: int, n_heads: int, dropout: float, ffn_dim: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        queries: Tensor,          # [B, Q, D]
        context: Tensor,          # [B, K, D]
        key_padding_mask: Tensor | None = None,  # [B, K] True=ignore (PyTorch convention)
    ) -> Tensor:
        # Cross-attention: queries attend to context
        x = queries
        attn_out, _ = self.cross_attn(x, context, context, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)

        # Self-attention among queries
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm2(x + attn_out)

        # FFN
        x = self.norm3(x + self.ffn(x))
        return x


class DAVIDEncoder(nn.Module):
    """Transformer encoder that compresses N visual tokens into L latent tokens.

    Uses cross-attention with L learned query vectors attending to the N input tokens.
    Outputs mu and logvar for the VAE reparameterization.
    """

    def __init__(self, config: DAVIDConfig):
        super().__init__()
        D = config.latent_dim
        ffn_dim = D * config.ffn_multiplier

        # L learned query vectors (broadcast over batch)
        self.queries = nn.Parameter(torch.randn(1, config.L, D) * 0.02)

        # Input projection in case input_dim != latent_dim
        if config.input_dim != config.latent_dim:
            self.input_proj = nn.Linear(config.input_dim, config.latent_dim)
        else:
            self.input_proj = nn.Identity()

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(D, config.n_heads, config.dropout, ffn_dim)
            for _ in range(config.n_encoder_layers)
        ])

        # Project to 2*latent_dim for mu + logvar
        self.out_proj = nn.Linear(D, 2 * D)

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            features: [B, N, input_dim] — padded visual features.
            mask:     [B, N] bool — True = valid token.

        Returns:
            mu, logvar: each [B, L, latent_dim]
        """
        B = features.shape[0]

        context = self.input_proj(features)  # [B, N, D]
        queries = self.queries.expand(B, -1, -1)  # [B, L, D]

        # PyTorch MHA key_padding_mask: True means IGNORE that position
        # Our mask is True=valid, so invert it
        key_padding_mask = ~mask  # [B, N]

        for block in self.blocks:
            queries = block(queries, context, key_padding_mask=key_padding_mask)

        # Project to mu and logvar
        out = self.out_proj(queries)  # [B, L, 2*D]
        mu, logvar = out.chunk(2, dim=-1)  # each [B, L, D]

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -10.0, 4.0)

        return mu, logvar


class DAVIDDecoder(nn.Module):
    """Transformer decoder that reconstructs N_queries tokens from m latent tokens.

    Uses cross-attention with N_queries learned positional queries attending to
    the (possibly truncated) latent prefix z[:, :m, :].
    """

    def __init__(self, config: DAVIDConfig):
        super().__init__()
        D = config.latent_dim
        ffn_dim = D * config.ffn_multiplier

        # N_queries learned positional query vectors
        self.queries = nn.Parameter(torch.randn(1, config.N_queries, D) * 0.02)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(D, config.n_heads, config.dropout, ffn_dim)
            for _ in range(config.n_decoder_layers)
        ])

        # Project back to input_dim (feature space)
        self.out_proj = nn.Linear(D, config.input_dim)

    def forward(self, z_prefix: Tensor) -> Tensor:
        """
        Args:
            z_prefix: [B, m, latent_dim] — truncated latent prefix (m <= L).

        Returns:
            recon: [B, N_queries, input_dim] — reconstructed features.
        """
        B = z_prefix.shape[0]
        queries = self.queries.expand(B, -1, -1)  # [B, N_queries, D]

        # No key_padding_mask needed — all m latent tokens are valid
        for block in self.blocks:
            queries = block(queries, z_prefix, key_padding_mask=None)

        recon = self.out_proj(queries)  # [B, N_queries, input_dim]
        return recon


class DAVIDVAE(nn.Module):
    """Full DAVID VAE: encoder + reparameterization + stochastic prefix truncation + decoder.

    During training, a random prefix length m is sampled each forward pass, forcing
    the model to encode coarse information in early tokens and fine details in later ones.
    At inference, all L tokens are used (or an arbitrary prefix for adaptive reasoning).
    """

    def __init__(self, config: DAVIDConfig):
        super().__init__()
        self.config = config
        self.encoder = DAVIDEncoder(config)
        self.decoder = DAVIDDecoder(config)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z ~ N(mu, exp(logvar)) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        features: Tensor,
        mask: Tensor,
        training: bool = True,
    ) -> DAVIDOutput:
        """
        Args:
            features: [B, N, D] — padded visual features from backbone.
            mask:     [B, N] bool — True = valid token.
            training: If True, apply stochastic prefix truncation.

        Returns:
            DAVIDOutput with recon, mu, logvar, m.
        """
        # Encode
        mu, logvar = self.encoder(features, mask)  # each [B, L, D]

        # Reparameterize
        z = self.reparameterize(mu, logvar)  # [B, L, D]

        # Stochastic prefix truncation
        L = self.config.L
        if training:
            m = torch.randint(1, L + 1, (1,)).item()
        else:
            m = L
        z_prefix = z[:, :m, :]  # [B, m, D]

        # Decode
        recon = self.decoder(z_prefix)  # [B, N_queries, input_dim]

        return DAVIDOutput(recon=recon, mu=mu, logvar=logvar, m=m)

    def encode(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return mu, logvar without sampling."""
        return self.encoder(features, mask)

    def decode(self, z_prefix: Tensor) -> Tensor:
        """Decode from a latent prefix of any length <= L."""
        return self.decoder(z_prefix)
