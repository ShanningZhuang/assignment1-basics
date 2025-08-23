import torch
import torch.nn as nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.

        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.

        Note: Remember to upcast your input to torch.float32 before performing
        the normalization (and later downcast to the original dtype).
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        result = einsum(x, self.gain, "... d_model, d_model -> ... d_model") / RMS

        # Return the result in the original dtype
        return result.to(in_dtype)
