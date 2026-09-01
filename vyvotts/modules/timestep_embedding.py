import math

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding using sinusoidal encoding followed by MLP projection.

    Based on the original Transformer positional encoding and diffusion models.
    Converts scalar timesteps to high-dimensional embeddings.

    Args:
        embedding_dim: Dimension of sinusoidal embedding (typically 256)
        hidden_dim: Dimension to project to (model hidden dimension)
        max_period: Maximum period for sinusoidal encoding (default: 10000)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 1024,
        max_period: int = 10000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

        # MLP projection: sinusoidal -> hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape [B] or [B, 1] containing timesteps in [0, 1]

        Returns:
            Embeddings of shape [B, hidden_dim]
        """
        # Ensure timesteps is 1D
        if timesteps.ndim > 1:
            timesteps = timesteps.squeeze(-1)

        # Sinusoidal embedding
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Project to hidden dimension
        emb = self.mlp(emb)

        return emb


def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000
) -> torch.Tensor:
    """
    Standalone function for sinusoidal timestep embedding.

    Args:
        timesteps: Tensor of shape [B] containing timesteps
        embedding_dim: Dimension of the embedding
        max_period: Maximum period for sinusoidal encoding

    Returns:
        Embeddings of shape [B, embedding_dim]
    """
    half_dim = embedding_dim // 2
    emb = math.log(max_period) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    return emb
