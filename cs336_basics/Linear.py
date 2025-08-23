import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from einops import einsum


class Linear(nn.Module):
    """
    Linear transformation module that performs y = xW^T without bias.

    This implementation follows the interface of PyTorch's nn.Linear module
    but without bias support.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Construct a linear transformation module.

        Args:
            in_features (int): Size of each input sample (final dimension)
            out_features (int): Size of each output sample (final dimension)
            device (torch.device, optional): Device to store the parameters on
            dtype (torch.dtype, optional): Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the weight parameter using truncated normal distribution."""
        mean = 0
        std = np.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.

        Args:
            x (torch.Tensor): Input tensor with shape (..., in_features)

        Returns:
            torch.Tensor: Output tensor with shape (..., out_features)
        """

        return einsum(
            x,
            self.weight,
            "... in_features, out_features in_features -> ... out_features",
        )
