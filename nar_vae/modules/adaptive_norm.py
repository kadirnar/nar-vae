import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN).

    Modulates layer normalization with scale and shift parameters
    derived from conditioning signals (e.g., timestep embeddings).

    Args:
        normalized_shape: Shape of the input to normalize (e.g., hidden_dim)
        conditioning_dim: Dimension of the conditioning vector
        eps: Epsilon for numerical stability
    """

    def __init__(
        self,
        normalized_shape: int,
        conditioning_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)

        # Linear layer to produce scale and shift from conditioning
        self.linear = nn.Linear(conditioning_dim, normalized_shape * 2)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, D] or [B, D, T]
            conditioning: Conditioning tensor of shape [B, conditioning_dim]

        Returns:
            Modulated output of same shape as x
        """
        # Normalize
        x_normalized = self.norm(x)

        # Generate scale and shift
        params = self.linear(conditioning)
        scale, shift = params.chunk(2, dim=-1)

        # Handle different input shapes
        if x.ndim == 3 and x.shape[1] != conditioning.shape[1]:
            # Input is [B, T, D], expand conditioning to [B, 1, D]
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        # Apply modulation: scale * x + shift
        return scale * x_normalized + shift


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization (AdaLN-Zero).

    Similar to AdaLN but also produces gate parameters for residual connections,
    initialized to zero for stable training. Used in DiT architecture.

    Args:
        normalized_shape: Shape of the input to normalize
        conditioning_dim: Dimension of the conditioning vector
        num_outputs: Number of output modulation parameters (typically 6 for DiT:
                     scale1, shift1, gate1, scale2, shift2, gate2)
        eps: Epsilon for numerical stability
    """

    def __init__(
        self,
        normalized_shape: int,
        conditioning_dim: int,
        num_outputs: int = 6,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)

        # Linear layer to produce modulation parameters
        self.linear = nn.Linear(conditioning_dim, normalized_shape * num_outputs)

        # Zero initialization for stability
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.num_outputs = num_outputs
        self.normalized_shape = normalized_shape

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape [B, T, D]
            conditioning: Conditioning tensor of shape [B, conditioning_dim]

        Returns:
            Tuple of (normalized_x, modulation_params)
            - normalized_x: Normalized input [B, T, D]
            - modulation_params: List of modulation tensors
        """
        # Normalize
        x_normalized = self.norm(x)

        # Generate modulation parameters
        params = self.linear(conditioning)  # [B, D * num_outputs]
        params = params.reshape(
            params.shape[0], self.num_outputs, self.normalized_shape
        )  # [B, num_outputs, D]

        # Expand for sequence dimension if needed
        if x.ndim == 3:
            params = params.unsqueeze(1)  # [B, 1, num_outputs, D]

        # Split into individual modulation parameters
        modulation_params = [params[:, :, i, :] for i in range(self.num_outputs)]

        return x_normalized, modulation_params


class ModulatedLayerNorm(nn.Module):
    """
    Layer normalization with modulation, combining normalization and affine transformation.

    This is a simpler variant that directly applies scale and shift from conditioning.

    Args:
        dim: Feature dimension
        conditioning_dim: Conditioning dimension
        eps: Epsilon for LayerNorm
    """

    def __init__(self, dim: int, conditioning_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.scale_shift = nn.Linear(conditioning_dim, dim * 2, bias=True)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] or [B, D]
            conditioning: [B, C]

        Returns:
            Modulated output same shape as x
        """
        # Get scale and shift
        scale_shift = self.scale_shift(conditioning)
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Normalize
        x = self.norm(x)

        # Apply modulation (handle both 2D and 3D inputs)
        if x.ndim == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        return x * (1 + scale) + shift
