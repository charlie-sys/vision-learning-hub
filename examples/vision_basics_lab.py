"""Self-contained PyTorch lab for the companion Chinese vision tutorial.

Run: python vision_basics_lab.py
No external dataset or pretrained weights are required.
"""

import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n: int, seed: int) -> TensorDataset:
    """Classify whether a bright square is on the left or right."""
    g = torch.Generator().manual_seed(seed)
    x = 0.15 * torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    row = torch.randint(5, 18, (n,), generator=g)
    for i in range(n):
        col = 4 if int(y[i]) == 0 else 18
        x[i, 0, int(row[i]) : int(row[i]) + 6, col : col + 6] += 1.0
    return TensorDataset(x.clamp(0, 1), y)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_demo():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(make_data(800, 11), batch_size=64, shuffle=True)
    val = DataLoader(make_data(200, 22), batch_size=64)
    model = TinyCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(3):
        model.train()
        loss_sum = 0.0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for x, y in val:
                pred = model(x.to(device)).argmax(dim=1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
        print(f"epoch={epoch + 1} train_loss={loss_sum / len(train.dataset):.4f} val_acc={correct / total:.3f}")
    with tempfile.TemporaryDirectory() as temp_dir:
        out = Path(temp_dir) / "tinycnn_state.pt"
        torch.save(model.state_dict(), out)
        restored = TinyCNN().to(device)
        restored.load_state_dict(torch.load(out, map_location=device, weights_only=True))
        restored.eval()
        with torch.inference_mode():
            probe = next(iter(val))[0][:2].to(device)
            assert torch.allclose(model(probe), restored(probe))
    print(f"checkpoint round-trip passed; device={device}")


def concepts_demo():
    # Q, K, V: [batch, tokens, channels]. Attention output has the same shape.
    tokens = torch.randn(2, 4, 8)
    q, k, v = tokens, tokens, tokens
    weights = torch.softmax((q @ k.transpose(-2, -1)) / (8 ** 0.5), dim=-1)
    attended = weights @ v
    assert weights.shape == (2, 4, 4) and attended.shape == tokens.shape
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4))

    # VAE: sample a latent variable while retaining gradients to mu and logvar.
    mu = torch.zeros(3, 5, requires_grad=True)
    logvar = torch.zeros(3, 5, requires_grad=True)
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    (z.square().mean() + kl).backward()
    assert mu.grad is not None and logvar.grad is not None

    # DDPM forward process at an arbitrary timestep; not a full diffusion trainer.
    x0 = torch.randn(4, 1, 28, 28)
    alpha_bar_t = torch.tensor([0.99, 0.8, 0.5, 0.1]).view(4, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * eps
    assert xt.shape == x0.shape
    print("attention, VAE reparameterization, DDPM noising: shape/gradient checks passed")


if __name__ == "__main__":
    concepts_demo()
    train_demo()
