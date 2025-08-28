from collections.abc import Callable, Iterable
from typing import Optional
import math
import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    l2_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
    if l2_norm < max_l2_norm:
        return
    else:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * (max_l2_norm / (l2_norm + eps))


class SGD(torch.optim.Optimizer):
    """Stochastic Gradient Descent optimizer with learning rate schedule.

    This implementation uses a learning rate schedule that decays as 1/sqrt(t+1)
    where t is the iteration number.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: 1e-3.

    Raises:
        ValueError: If learning rate is negative.
    """

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay.

    Implements the AdamW algorithm with bias correction and weight decay.
    The weight decay is applied directly to the parameters (decoupled from the gradient).

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        weight_decay (float): Weight decay coefficient (L2 penalty).
        betas (tuple[float, float]): Coefficients used for computing running averages
            of gradient and its square. Default: (0.9, 0.999).
        eps (float): Term added to the denominator to improve numerical stability.
            Default: 1e-8.

    Raises:
        ValueError: If learning rate is negative, betas are not in [0, 1), eps is negative,
            or weight_decay is negative.
    """

    def __init__(self, params, lr, weight_decay, betas, eps):
        # Parameter validation
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "lamb": weight_decay,
        }
        # Remove incorrect initialization of self.m and self.v
        # These should be per-parameter state, not global optimizer state
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lamb = group["lamb"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m = state["m"]
                v = state["v"]
                t = state["t"]
                t += 1
                state["t"] = t
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                # Apply weight decay (decoupled from gradient)
                p.data.mul_(1 - alpha * lamb)
                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1**t
                # Compute bias-corrected second raw moment estimate
                bias_correction2 = 1 - beta2**t
                # Update parameters
                p.data.addcdiv_(
                    m,
                    v.sqrt().add_(eps),
                    value=-alpha / bias_correction1 * math.sqrt(bias_correction2),
                )
        return loss
