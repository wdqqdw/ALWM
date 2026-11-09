"""Fixed-length scene compressors used by ALWM Stage 1."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _validate_inputs(features: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> None:
    if features.ndim != 3:
        raise ValueError(f"features must be [B,N,D], got {tuple(features.shape)}")
    if times.shape != features.shape[:2] or mask.shape != features.shape[:2]:
        raise ValueError("times and mask must have shape [B,N]")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("each sample must contain at least one valid visual token")


class TokenLearnerCompressor(nn.Module):
    """C1: eight content-adaptive attention maps with a weak temporal prior."""

    def __init__(self, hidden_size: int, latent_tokens: int = 8, bottleneck_dim: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_tokens = latent_tokens
        self.norm = nn.LayerNorm(hidden_size)
        self.score = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, latent_tokens),
        )
        self.refine = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_size),
        )
        self.temporal_strength = nn.Parameter(torch.tensor(1.0))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_inputs(features, times, mask)
        logits = self.score(self.norm(features)).transpose(1, 2)
        anchors = torch.linspace(0.0, 1.0, self.latent_tokens, device=features.device, dtype=features.dtype)
        prior = -torch.abs(times[:, None, :] - anchors[None, :, None])
        logits = logits + F.softplus(self.temporal_strength) * prior
        logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
        weights = F.softmax(logits.float(), dim=-1).to(features.dtype)
        pooled = torch.einsum("bkn,bnd->bkd", weights, features)
        return pooled + self.residual_scale * self.refine(pooled)


def build_compressor(
    name: str,
    hidden_size: int,
    latent_tokens: int = 8,
    bottleneck_dim: int = 1024,
) -> nn.Module:
    if name.lower() != "c1":
        raise ValueError("this release supports the C1 compressor")
    return TokenLearnerCompressor(hidden_size, latent_tokens, bottleneck_dim)
