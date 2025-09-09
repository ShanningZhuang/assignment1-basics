"""
Normalization Layers: LayerNorm and RMSNorm

Background Knowledge:
Normalization techniques are crucial for stabilizing training in deep neural networks,
particularly in Transformers. They help with gradient flow and convergence.

1. LayerNorm (Layer Normalization):
   - Normalizes inputs across the features dimension (not across batch like BatchNorm)
   - Forces each input to have zero mean and unit variance before rescaling
   - Widely adopted in Transformers (e.g., GPT, BERT)
   
   Formula: y = (x - μ) / √(σ² + ε) * γ + β
   where:
   - x: input vector (for each token or sample)
   - μ: mean of the features
   - σ²: variance of the features
   - ε: small constant for numerical stability
   - γ, β: learnable scale and shift parameters

2. RMSNorm (Root Mean Square Normalization):
   - Simplified variant of LayerNorm that avoids computing the mean
   - Normalizes the root mean square (RMS) of activations
   - Computationally cheaper and empirically works as well as LayerNorm for deep transformers
   - Used in modern LLMs like LLaMA, T5 variants
   
   Formula: y = x / √(1/d * Σx²ᵢ + ε) * γ
   where:
   - d: dimensionality of the feature vector
   - No centering (no subtraction of mean)
   - Only a scale parameter γ (typically no bias β)

Key Differences:
- LayerNorm: ensures zero mean & unit variance
- RMSNorm: skips mean-centering, normalizes only by magnitude (faster and simpler)
"""

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

        RMSNorm Formula: y = x / √(1/d * Σx²ᵢ + ε) * γ

        Note: Remember to upcast your input to torch.float32 before performing
        the normalization (and later downcast to the original dtype).
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Upcast to float32 for numerical stability

        # Compute RMS: √(1/d * Σx²ᵢ + ε) = √(mean(x²) + ε)
        # This corresponds to the denominator in the RMSNorm formula
        RMS = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps) # batch_size, sequence_length, 1
        
        # Apply normalization: x / RMS * γ (where γ is self.gain)
        # This implements: y = x / √(1/d * Σx²ᵢ + ε) * γ
        result = einsum(x, self.gain, "... d_model, d_model -> ... d_model") / RMS # batch_size, sequence_length, d_model

        # Return the result in the original dtype
        return result.to(in_dtype)

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the LayerNorm module.

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
        # γ (gain/scale parameter) - learnable parameter for scaling
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        # β (bias/shift parameter) - learnable parameter for shifting
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.

        LayerNorm Formula: y = (x - μ) / √(σ² + ε) * γ + β
        where:
        - μ: mean of the features (computed across d_model dimension)
        - σ²: variance of the features (computed across d_model dimension)
        - γ: learnable scale parameter (self.gain)
        - β: learnable shift parameter (self.bias)

        Note: Remember to upcast your input to torch.float32 before performing
        the normalization (and later downcast to the original dtype).
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Upcast to float32 for numerical stability

        # Compute mean (μ) across the feature dimension (d_model)
        # Shape: (batch_size, sequence_length, 1)
        mean = torch.mean(x, dim=-1, keepdim=True)
        
        # Center the input: (x - μ)
        x_centered = x - mean
        
        # Compute variance (σ²) across the feature dimension
        # Shape: (batch_size, sequence_length, 1)
        variance = torch.mean(torch.square(x_centered), dim=-1, keepdim=True)
        
        # Normalize: (x - μ) / √(σ² + ε)
        x_normalized = x_centered / torch.sqrt(variance + self.eps)
        
        # Apply scale and shift: * γ + β
        # This implements: y = (x - μ) / √(σ² + ε) * γ + β
        result = einsum(x_normalized, self.gain, "... d_model, d_model -> ... d_model") + self.bias

        # Return the result in the original dtype
        return result.to(in_dtype)


class NoNorm(nn.Module):
    """
    Identity layer that does nothing - for ablation studies to remove normalization.
    This allows us to test the impact of layer normalization by completely removing it.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Args:
            d_model: int Hidden dimension of the model (unused, for compatibility)
            eps: float = 1e-5 Epsilon value (unused, for compatibility)
            device: torch.device | None = None Device (unused, for compatibility)
            dtype: torch.dtype | None = None Data type (unused, for compatibility)
        """
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Identity function - returns input unchanged.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            The same tensor unchanged
        """
        return x
