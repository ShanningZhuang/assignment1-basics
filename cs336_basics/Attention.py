import torch
import torch.nn as nn
from einops import einsum, rearrange
import numpy as np
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding
from .Linear import Linear


def softmax(x: torch.Tensor, dim: int):
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        x (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `x` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `x` with the output of
        softmax normalizing the specified `dim`.
    """
    x_max = torch.max(x, dim, keepdim=True).values
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return output


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    qk_value = einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    ) / np.sqrt(
        Q.shape[-1]
    )  # ... queries keys
    if mask is not None:
        qk_value = qk_value.masked_fill(~mask, float("-inf"))

    # do softmax in keys dimension keeps queries
    qk_softmax = softmax(qk_value, dim=-1)  # ... queries keys
    attention = einsum(
        qk_softmax, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )  # ... queries d_v
    return attention


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            use_rope (bool): Whether to use RoPE
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
            token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = (
            RotaryPositionalEmbedding(theta, d_k=self.d_k, max_seq_len=max_seq_len)
            if use_rope
            else None
        )
        # Linear have no dimension, only input and output
        self.q_proj = Linear(d_model, self.num_heads * self.d_k)  # in d_model out h*d_q
        self.k_proj = Linear(d_model, self.num_heads * self.d_k)  # in d_model out h*d_k
        self.v_proj = Linear(d_model, self.num_heads * self.d_k)  # in d_model out h*d_v
        self.o_proj = Linear(self.num_heads * self.d_k, d_model)  # in h*d_v out d_model

    def forward(
        self, in_features: torch.Tensor, token_positions: torch.Tensor | None = None
    ):
        """
        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run implementation on.
            token_positions ... sequence_length
        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
        """
        qkv_proj = torch.cat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]
        )
        qkv = in_features @ qkv_proj.T
        queries, keys, values = qkv.chunk(3, -1)  # ... sequence_length h*d_q

        queries = rearrange(
            queries,
            "... sequence_length (h d_q) -> ... h sequence_length d_q",
            h=self.num_heads,
        )
        keys = rearrange(
            keys,
            "... sequence_length (h d_q) -> ... h sequence_length d_q",
            h=self.num_heads,
        )
        values = rearrange(
            values,
            "... sequence_length (h d_q) -> ... h sequence_length d_q",
            h=self.num_heads,
        )
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(queries.shape[-2]).view(
                    1, -1
                )  # 1 sequence_length
            token_positions.unsqueeze(-2)
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)
        seq_len = in_features.shape[-2]
        masks = (
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            .bool()
            .view(1, 1, seq_len, seq_len)
        )  # ... 1 seq_len seq_len, corresponding to Q, K, Q is row and K is column
        scores = scaled_dot_product_attention(
            queries, keys, values, ~masks
        )  # ... h sequence_length d_v
        scores = rearrange(
            scores,
            "... h sequence_length d_v -> ... sequence_length (h d_v)",
            h=self.num_heads,
        )
        output = self.o_proj(scores)
        return output
