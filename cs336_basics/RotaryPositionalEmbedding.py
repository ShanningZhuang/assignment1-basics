import torch
import torch.nn as nn
from einops import rearrange, repeat


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) module.
    RoPE injects positional information into a model by rotating pairs of embedding
    elements by an angle that depends on the token's position.
    Reference:
        Su et al., 2021 (https://arxiv.org/abs/2104.09864)
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Initializes the RoPE module.
        Args:
            theta (float): The base for the rotational angles. Corresponds to Θ in the paper.
            d_k (int): The dimension of the query and key vectors.
            max_seq_len (int): The maximum sequence length that the model will see.
            device (torch.device | None, optional): The device to store pre-computed
                                                    sin/cos values on. Defaults to None.
        """
        super().__init__()
        # Your implementation here
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        theta_i = torch.pow(theta, -2 * torch.arange(d_k / 2) / d_k).view(1, -1)
        cos_values = torch.cos(theta_i * torch.arange(max_seq_len).view(-1, 1))
        sin_values = torch.sin(theta_i * torch.arange(max_seq_len).view(-1, 1))
        self.register_buffer("cos_cached", cos_values, persistent=False)
        self.register_buffer("sin_cached", sin_values, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k). Note that this
                              module should tolerate an arbitrary number of batch dimensions.
            token_positions (torch.Tensor): A tensor of shape (..., seq_len) that specifies
                                            the token position for each vector in x.
        Returns:
            torch.Tensor: The input tensor with rotary positional embeddings applied.
                          The output shape is the same as the input shape.
        """
        # Your implementation here
        # cos_cached (seq_len, d_k)
        """
        x'_a = x_a * cos(θ) - x_b * sin(θ) odd d_k index
        x'_b = x_a * sin(θ) + x_b * cos(θ) even d_k index
        theta (token, d)
        
        Ri1 0 0 . . . 0
        0 Ri2 0 . . . 0
        0 0 Ri3 . . . 0
        ... ... ... ... ...
        0 0 0 . . . Rid/2
        """
        # The function signature allows for an arbitrary number of batch dimensions '...',
        # but token_positions might be passed in without them (e.g., shape [seq_len]).
        # The following logic robustly handles this by adding and expanding the
        # missing dimensions to match the input tensor x.

        # We subtract 1 because token_positions doesn't have the final d_k dimension.
        num_missing_dims = x.ndim - token_positions.ndim - 1
        if num_missing_dims > 0:
            # Add singleton dimensions for the missing batch dimensions.
            view_shape = (1,) * num_missing_dims + token_positions.shape
            token_positions = token_positions.view(view_shape)
            # Expand the singleton dimensions to match x's batch dimensions.
            expand_shape = x.shape[:-2] + (-1,)
            token_positions = token_positions.expand(expand_shape)

        cos_values = self.cos_cached[token_positions]
        sin_values = self.sin_cached[token_positions]

        # 2. Reshape x to isolate pairs using einops
        # Splits the last dimension d_k into two parts: d_k/2 and 2
        x_reshaped = rearrange(x, "... s (d2 p) -> ... s d2 p", p=2, d2=self.d_k // 2)
        x_odd, x_even = x_reshaped[..., 0], x_reshaped[..., 1]
        x_rotate_odd = x_odd * cos_values - x_even * sin_values
        x_rotate_even = x_odd * sin_values + x_even * cos_values
        stacked_pairs = torch.stack([x_rotate_odd, x_rotate_even], dim=-1)
        # Shape of stacked_pairs: (..., seq_len, d_k / 2, 2)
        # The pattern '... s d2 p -> ... s (d2 p)' is the exact inverse
        # of the pattern you used to split the tensor.
        output = rearrange(stacked_pairs, "... s d2 p -> ... s (d2 p)")
        # Shape of output: (..., seq_len, d_k)

        return output
