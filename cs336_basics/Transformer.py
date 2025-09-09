import torch
import torch.nn as nn
from .RMSNorm import RMSNorm, NoNorm
from .Attention import MultiheadSelfAttention, softmax
from .PositionwiseFeedForward import PositionwiseFeedForward, PositionwiseFeedForwardSiLU
from .Embedding import Embedding
from .Linear import Linear


class Transformer(nn.Module):
    """
    Pre-Norm Transformer Block
    
    Pre-Norm applies normalization BEFORE the sub-layers (attention and FFN).
    This is the modern approach used in many recent models like GPT, LLaMA, etc.
    
    Pre-Norm Equations:
    z = x + MultiHeadedSelfAttention(RMSNorm(x))
    y = z + FFN(RMSNorm(z))
    """
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
        norm_type: str = "rmsnorm", use_rope: bool = True, ffn_type: str = "swiglu"
    ):
        """
        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache.
            theta (float): RoPE parameter.
            norm_type (str): Type of normalization ("rmsnorm" or "nonorm").
            use_rope (bool): Whether to use rotary position embeddings.
            ffn_type (str): Type of feedforward network ("swiglu" or "silu").
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.norm_type = norm_type
        self.use_rope = use_rope
        self.ffn_type = ffn_type
        
        # Choose normalization based on config
        if norm_type == "rmsnorm":
            self.rms_norm_mha = RMSNorm(d_model)
            self.rms_norm_ffn = RMSNorm(d_model)
        elif norm_type == "nonorm":
            self.rms_norm_mha = NoNorm(d_model)
            self.rms_norm_ffn = NoNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}. Use 'rmsnorm' or 'nonorm'.")
            
        self.multi_head_self_attention = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        # Choose feedforward network based on config
        if ffn_type == "swiglu":
            self.positionwise_feedforward = PositionwiseFeedForward(
                d_model=d_model, d_ff=d_ff
            )
        elif ffn_type == "silu":
            # For SiLU, use 4*d_model to approximately match parameter count
            # SwiGLU has 3 weight matrices, SiLU has 2, so we increase d_ff
            d_ff_silu = 4 * d_model
            self.positionwise_feedforward = PositionwiseFeedForwardSiLU(
                d_model=d_model, d_ff=d_ff_silu
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}. Use 'swiglu' or 'silu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-Norm Transformer Forward Pass
        
        Implements the Pre-Norm equations:
        z = x + MultiHeadedSelfAttention(RMSNorm(x))
        y = z + FFN(RMSNorm(z))
        
        Input:
        x batch seq_len d_model
        Returns:
            Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
            running the Transformer block on the input features while using RoPE.
        """
        # First residual connection: z = x + MultiHeadedSelfAttention(RMSNorm(x))
        z = x + self.multi_head_self_attention(self.rms_norm_mha(x))
        
        # Second residual connection: y = z + FFN(RMSNorm(z))
        y = z + self.positionwise_feedforward(self.rms_norm_ffn(z))
        
        return y


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
        norm_type: str = "rmsnorm",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
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
            norm_type (str): Type of normalization ("rmsnorm" or "nonorm").
            use_rope (bool): Whether to use rotary position embeddings.
            ffn_type (str): Type of feedforward network ("swiglu" or "silu").
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
                    norm_type=norm_type,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
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
        x = self.input_embedding(in_indices) # batch_size, sequence_length, embedding_dim
        x = self.transformer(x)
        x = self.rms_norm(x)
        x = self.output_embedding(x)
        # output = softmax(x, dim=-1)
        return x


class TransformerPostNorm(nn.Module):
    """
    Post-Norm Transformer Block
    
    Post-Norm applies normalization AFTER the sub-layers (attention and FFN).
    This is the original approach from "Attention is All You Need" paper.
    
    Post-Norm Equations:
    z = RMSNorm(x + MultiHeadedSelfAttention(x))
    y = RMSNorm(z + FFN(z))
    """
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
        Post-Norm Transformer Forward Pass
        
        Implements the Post-Norm equations:
        z = RMSNorm(x + MultiHeadedSelfAttention(x))
        y = RMSNorm(z + FFN(z))
        
        Input:
        x batch seq_len d_model
        Returns:
            Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
            running the Transformer block on the input features while using RoPE.
        """
        # First residual connection with post-norm: z = RMSNorm(x + MultiHeadedSelfAttention(x))
        z = self.rms_norm_mha(x + self.multi_head_self_attention(x))
        
        # Second residual connection with post-norm: y = RMSNorm(z + FFN(z))
        y = self.rms_norm_ffn(z + self.positionwise_feedforward(z))
        
        return y
    
class TransformerLMPostNorm(nn.Module):
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
                TransformerPostNorm(
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
        x = self.input_embedding(in_indices) # batch_size, sequence_length, embedding_dim
        x = self.transformer(x)
        x = self.rms_norm(x)
        x = self.output_embedding(x)
        # output = softmax(x, dim=-1)
        return x