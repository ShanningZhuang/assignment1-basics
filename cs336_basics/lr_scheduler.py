import math


class CosineScheduler:
    """
    Cosine learning rate scheduler with linear warmup.

    Args:
        alpha_max: Maximum learning rate
        alpha_min: Minimum learning rate
        Tw: Warmup time steps
        Tc: Total time steps for cosine annealing
    """

    def __init__(self, alpha_max, alpha_min, Tw, Tc):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.Tw = Tw
        self.Tc = Tc

    def get_lr(self, t):
        """
        Get learning rate at time step t.

        Args:
            t: Current time step

        Returns:
            Learning rate at time step t
        """
        if t < self.Tw:
            alpha_t = self.alpha_max * t / self.Tw
        elif self.Tw <= t <= self.Tc:
            alpha_t = self.alpha_min + 0.5 * (
                1 + math.cos((t - self.Tw) / (self.Tc - self.Tw) * math.pi)
            ) * (self.alpha_max - self.alpha_min)
        else:
            alpha_t = self.alpha_min
        return alpha_t
