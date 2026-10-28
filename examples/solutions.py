"""Reference solutions to the course's tensor and numerical exercises."""
import torch
from torch import nn


class ProjectedAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, allowed=None):
        scores = self.q(x) @ self.k(x).transpose(-2, -1) / x.shape[-1] ** 0.5
        if allowed is not None:
            if not allowed.any(-1).all():
                raise ValueError("Each query needs at least one allowed key")
            scores = scores.masked_fill(~allowed, -torch.inf)
        weights = scores.softmax(-1)
        return self.out(weights @ self.v(x)), weights


def scalar_gradient(w, x=2.0, b=1.0, target=5.0):
    return (w * x + b - target) ** 2 / 2


def info_nce(scores, positive_index, temperature=0.5):
    return nn.functional.cross_entropy(scores / temperature, positive_index)


def euler_decay(step, time=1.0):
    n = round(time / step)
    if abs(n * step - time) > 1e-8:
        raise ValueError("time must be a multiple of step")
    x = 1.0
    for _ in range(n):
        x -= step * x
    return x


def brownian_terminal(n=10000, steps=100, time=1.0, seed=7):
    generator = torch.Generator().manual_seed(seed)
    increments = torch.randn(n, steps, generator=generator) * (time / steps) ** 0.5
    return increments.sum(1)


if __name__ == "__main__":
    torch.manual_seed(7)
    model = ProjectedAttention()
    output, weights = model(torch.randn(2, 4, 8))
    output.square().mean().backward()
    print("Attention output shape:", tuple(output.shape))
    print("Rows sum to one:", torch.allclose(weights.sum(-1), torch.ones(2, 4)))
    print("Projection gets gradient:", model.q.weight.grad.norm().item())
    w = torch.tensor(3.0, requires_grad=True)
    scalar_gradient(w).backward()
    print("Scalar gradient (expected 4):", w.grad.item())
    print("Brownian variance (approximately 1):", brownian_terminal().var().item())
