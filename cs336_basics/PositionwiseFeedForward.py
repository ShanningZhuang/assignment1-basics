import torch
import torch.nn as nn
from .Linear import Linear


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Initializes the PositionwiseFeedForward module using SwiGLU.

        Args:
            d_model (int): Dimensionality of the feedforward input and output
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu
            w1 (Float[Tensor, "d_ff d_model"]): Stored weights for W1
            w2 (Float[Tensor, "d_model d_ff"]): Stored weights for W2
            w3 (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the PositionwiseFeedForward module.
        This should implement the SwiGLU activation function.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.w2(SiLU(self.w1(x)) * self.w3(x))
