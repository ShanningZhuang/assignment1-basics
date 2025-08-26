import torch
import torch.nn as nn
from .RMSNorm import RMSNorm
from .Attention import MultiheadSelfAttention, softmax
from .PositionwiseFeedForward import PositionwiseFeedForward
from .Embedding import Embedding
from .Linear import Linear


class Transformer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        """
        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache.
            theta (float): RoPE parameter.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms_norm_mha = RMSNorm(d_model)
        self.multi_head_self_attention = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.rms_norm_ffn = RMSNorm(d_model)

        self.positionwise_feedforward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
        x batch seq_len d_model
        Returns:
            Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
            running the Transformer block on the input features while using RoPE.
        """
        y = x + self.multi_head_self_attention(self.rms_norm_mha(x))
        output = y + self.positionwise_feedforward(self.rms_norm_ffn(y))
        return output


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        """
        Args:
            vocab_size (int): The number of unique items in the output vocabulary to be predicted.
            context_length (int): The maximum number of tokens to process at once.
            d_model (int): The dimensionality of the model embeddings and sublayer outputs.
            num_layers (int): The number of Transformer layers to use.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
            rope_theta (float): The RoPE $Theta$ parameter.
        """
        super().__init__()
        self.input_embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.transformer = nn.Sequential(
            *[
                Transformer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model=d_model)
        self.output_embedding = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor):
        """
        Args:
            in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on.
            Shape is (batch_size, sequence_length), where `sequence_length` is at most `context_length`.

        Returns:
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
            next-word distribution for each token.
        """
        x = self.input_embedding(in_indices)
        x = self.transformer(x)
        x = self.rms_norm(x)
        x = self.output_embedding(x)
        # output = softmax(x, dim=-1)
        return x
