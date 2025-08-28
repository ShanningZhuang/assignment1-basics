from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
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
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


def run_experiment(lr, iterations=10):
    """Run SGD optimization for given learning rate and return loss trajectory"""
    print(f"\n=== Learning Rate: {lr} ===")
    
    # Reset weights for each experiment
    torch.manual_seed(42)  # For reproducibility
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    
    losses = []
    
    for t in range(iterations):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        loss_value = loss.cpu().item()
        losses.append(loss_value)
        print(f"Iteration {t}: Loss = {loss_value:.6f}")
        
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
    
    return losses


if __name__ == "__main__":
    learning_rates = [1e1, 1e2, 1e3]
    
    print("Running SGD experiment with different learning rates...")
    print("=" * 60)
    
    all_results = {}
    
    for lr in learning_rates:
        losses = run_experiment(lr, iterations=10)
        all_results[lr] = losses
        
        # Analyze behavior
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        if final_loss > initial_loss:
            behavior = "DIVERGING"
        elif final_loss < initial_loss * 0.1:  # Significant decay
            behavior = "FAST DECAY"
        elif final_loss < initial_loss:
            behavior = "SLOW DECAY"
        else:
            behavior = "STABLE"
            
        print(f"Behavior: {behavior} (Initial: {initial_loss:.6f}, Final: {final_loss:.6f})")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    for lr in learning_rates:
        losses = all_results[lr]
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        if final_loss > initial_loss:
            behavior = "diverges (loss increases)"
        elif final_loss < initial_loss * 0.1:
            behavior = "decays very fast"
        elif final_loss < initial_loss:
            behavior = "decays slowly"
        else:
            behavior = "remains stable"
            
        print(f"LR {lr}: {behavior}")
