
## rolling mean & variance

class OnlineStats:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        """Update stats for new observation."""
        self.n += 1
        new_mean = self.mean + (x - self.mean) / self.n
        self.m2 += (x - self.mean) * (x - new_mean)
        self.mean = new_mean

    @property
    def var(self) -> float:
        if self.n > 1:
            return self.m2 / (self.n - 1)
        else:
            return 0.0

    @property
    def precision(self) -> float:
        """Inverse of variance."""
        if self.n > 1:
            return (self.n - 1) / max(self.m2, 1.0e-6)
        else:
            return 1.0

    @property
    def std(self) -> float:
        return self.var ** 0.5

    def __repr__(self):
        return f'<OnlineStats mean={self.mean} std={self.std}'
