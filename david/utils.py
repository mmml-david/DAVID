"""Utility functions for padding, masking, feature interpolation, and EMA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def pad_sequence_to_max(tensors: list[Tensor], pad_value: float = 0.0) -> tuple[Tensor, Tensor]:
    """Pad a list of [N_i, D] tensors to [batch, N_max, D] with a bool mask.

    Args:
        tensors: List of tensors with shape [N_i, D].
        pad_value: Value used for padding.

    Returns:
        padded: [batch, N_max, D]
        mask:   [batch, N_max] — True where token is valid (not padding)
    """
    batch = len(tensors)
    D = tensors[0].shape[-1]
    lengths = [t.shape[0] for t in tensors]
    N_max = max(lengths)

    padded = torch.full((batch, N_max, D), pad_value, dtype=tensors[0].dtype)
    mask = torch.zeros(batch, N_max, dtype=torch.bool)

    for i, (t, n) in enumerate(zip(tensors, lengths)):
        padded[i, :n] = t
        mask[i, :n] = True

    return padded, mask


def build_padding_mask(lengths: list[int], max_len: int, device=None) -> Tensor:
    """Build a bool mask [batch, max_len] — True = valid token."""
    batch = len(lengths)
    mask = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    return mask


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of all trainable parameters:
        shadow = decay * shadow + (1 - decay) * param

    Usage:
        ema = EMAModel(vae, decay=0.999)
        # after each optimizer.step():
        ema.update(vae)
        # to evaluate with EMA weights:
        ema.apply(vae)
        ... evaluate ...
        ema.restore(vae)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Call after every optimizer.step()."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Swap EMA weights into the model (saves originals for restore())."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore the original (non-EMA) weights after apply()."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]


def interpolate_features(features: Tensor, target_n: int) -> Tensor:
    """Interpolate [batch, N, D] → [batch, target_n, D] using linear interpolation.

    Treats D as channels and N as the spatial dimension for F.interpolate.
    """
    B, N, D = features.shape
    if N == target_n:
        return features
    # F.interpolate expects [B, C, L] for 1D
    x = features.permute(0, 2, 1)  # [B, D, N]
    x = F.interpolate(x, size=target_n, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)  # [B, target_n, D]
