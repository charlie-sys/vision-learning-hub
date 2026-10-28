"""Four complete small-scale vision labs. Run --help from the repository root.

Original educational implementations, not replicas of the papers' full systems.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import random
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DINO_REF = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"


class CourseFashionMNIST(datasets.FashionMNIST):
    # Same checksummed FashionMNIST files, retrieved from the dataset authors.
    mirrors = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]


class SyntheticImages(Dataset):
    """Deterministic class-position markers for offline smoke checks only."""
    def __init__(self, n, seed, transform):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.arange(n) % 10
        self.images = (torch.rand(n, 28, 28, generator=g) * 30).to(torch.uint8)
        for i, label in enumerate(self.labels):
            y, x = 3 + (int(label) // 5) * 13, 2 + (int(label) % 5) * 5
            self.images[i, y:y + 6, x:x + 4] = 230
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i].numpy())), self.labels[i]


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def data_splits(args):
    transform = transforms.ToTensor()
    if args.mode == "ddpm":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if args.mode == "probe":
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    if args.dataset == "synthetic":
        train = SyntheticImages(args.train_size, args.seed, transform)
        val = SyntheticImages(args.val_size, args.seed + 1, transform)
        return train, val, None, {"dataset": "synthetic smoke only", "test_used": False}
    full = CourseFashionMNIST(args.data_dir, train=True, download=True, transform=transform)
    indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(args.seed))
    if args.train_size > 50000 or args.val_size > 10000:
        raise ValueError("Use at most 50000 train and 10000 validation examples")
    train_idx, val_idx = indices[:args.train_size], indices[50000:50000 + args.val_size]
    train, val = Subset(full, train_idx.tolist()), Subset(full, val_idx.tolist())
    test = None
    if args.evaluate_test:
        test = CourseFashionMNIST(args.data_dir, train=False, download=True, transform=transform)
    return train, val, test, {
        "dataset": "FashionMNIST", "split_seed": args.seed,
        "train_indices_sha256": hashlib.sha256(train_idx.numpy().tobytes()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(val_idx.numpy().tobytes()).hexdigest(),
        "test_used": bool(args.evaluate_test),
    }


def loader(data, args, epoch=None):
    return DataLoader(data, batch_size=args.batch_size, shuffle=epoch is not None,
                      generator=torch.Generator().manual_seed(args.seed + (epoch or 0)), num_workers=0)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)


def gaussian_kl(mu, logvar):
    """KL(q||N(0,I)), sum over latent dimensions, mean over examples."""
    return (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)).mean()


def reparameterize(mu, logvar, generator=None):
    noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu + (0.5 * logvar).exp() * noise


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mu, self.logvar = nn.Linear(256, latent_dim), nn.Linear(256, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 784))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    def forward(self, x, generator=None):
        mu, lv = self.encode(x)
        return self.decode(reparameterize(mu, lv, generator)), mu, lv


class TimeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=64):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1, self.norm2 = nn.GroupNorm(8, out_ch), nn.GroupNorm(8, out_ch)
        self.time = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time):
        h = F.silu(self.norm1(self.conv1(x))) + self.time(time)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(h)) + self.skip(x))


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.down1, self.down2 = TimeBlock(1, 32), TimeBlock(32, 64)
        self.middle = TimeBlock(64, 64)
        self.up2, self.up1 = TimeBlock(128, 64), TimeBlock(96, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        frequencies = torch.exp(-math.log(10000) * torch.arange(32, device=x.device) / 31)
        angles = t.float()[:, None] * frequencies[None]
        time = self.time_mlp(torch.cat([angles.sin(), angles.cos()], 1))
        h1 = self.down1(x, time)
        h2 = self.down2(F.avg_pool2d(h1, 2), time)
        mid = self.middle(F.avg_pool2d(h2, 2), time)
        up2 = self.up2(torch.cat([F.interpolate(mid, size=h2.shape[-2:], mode="nearest"), h2], 1), time)
        up1 = self.up1(torch.cat([F.interpolate(up2, size=h1.shape[-2:], mode="nearest"), h1], 1), time)
        return self.out(up1)


class Diffusion:
    def __init__(self, steps, device):
        self.steps = steps
        self.beta = torch.linspace(1e-4, 0.05, steps, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(0)
        previous = torch.cat([torch.ones(1, device=device), self.alpha_bar[:-1]])
        self.posterior_var = self.beta * (1 - previous) / (1 - self.alpha_bar)

    def noise(self, x, t, eps):
        bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        return bar.sqrt() * x + (1 - bar).sqrt() * eps

    @torch.inference_mode()
    def sample(self, model, n, seed):
        g = torch.Generator(device=self.beta.device).manual_seed(seed)
        x = torch.randn((n, 1, 28, 28), generator=g, device=self.beta.device)
        model.eval()
        for i in reversed(range(self.steps)):
            t = torch.full((n,), i, device=x.device, dtype=torch.long)
            eps = model(x, t)
            x0 = ((x - (1 - self.alpha_bar[i]).sqrt() * eps) / self.alpha_bar[i].sqrt()).clamp(-1, 1)
            prev = self.alpha_bar[i - 1] if i else torch.ones((), device=x.device)
            mean = (prev.sqrt() * self.beta[i] / (1 - self.alpha_bar[i])) * x0
            mean += (self.alpha[i].sqrt() * (1 - prev) / (1 - self.alpha_bar[i])) * x
            x = mean
            if i:
                x = x + self.posterior_var[i].sqrt() * torch.randn(x.shape, generator=g, device=x.device)
        return ((x.clamp(-1, 1) + 1) / 2).cpu()


@torch.inference_mode()
def classification_eval(model, data, args):
    model.eval()
    loss_sum = correct = count = 0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for x, y in loader(data, args):
        logits = model(x.to(args.device))
        loss_sum += F.cross_entropy(logits, y.to(args.device), reduction="sum").item()
        pred = logits.argmax(1).cpu()
        correct += (pred == y).sum().item()
        confusion += torch.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
        count += len(y)
    return {"loss": loss_sum / count, "accuracy": correct / count, "confusion": confusion.tolist()}


def write_curve(history, output):
    """Dependency-free loss/accuracy plot; no implied quality thresholds."""
    keys = [key for key in history[0] if key != "epoch"]
    panels = []
    for index, key in enumerate(keys):
        vals = [row[key] for row in history]
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 1e-8)
        points = " ".join(f'{50 + 500 * i / max(len(vals)-1, 1):.1f},{95 - 65 * (v-lo) / span:.1f}' for i, v in enumerate(vals))
        panels.append(f'<g transform="translate(0,{index*130})"><text x="20" y="18">{key}: {vals[-1]:.5f}</text><line x1="50" y1="100" x2="550" y2="100" stroke="#ccd"/><polyline fill="none" stroke="#176e76" stroke-width="2" points="{points}"/><text x="50" y="120">epoch 1 → {history[-1]["epoch"]}</text></g>')
    output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="{130*len(keys)}" viewBox="0 0 600 {130*len(keys)}"><rect width="100%" height="100%" fill="white"/>' + ''.join(panels) + '</svg>', encoding="utf-8")


def save_checkpoint(path, model, opt, epoch, history, args):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                "history": history, "config": vars(args), "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def restore(model, opt, args):
    if not args.resume:
        return 0, []
    ckpt = torch.load(args.resume, map_location=args.device, weights_only=True)
    for key in ("mode", "dataset", "seed", "train_size", "val_size", "latent_dim", "beta", "steps", "batch_size", "lr"):
        if ckpt["config"].get(key) != getattr(args, key, None):
            raise ValueError(f"Resume configuration differs: {key}")
    model.load_state_dict(ckpt["model"])
    if Path(ckpt["config"]["output"]).resolve() != Path(args.output).resolve():
        raise ValueError("Resume in the same output directory to retain best.pt")
    if args.epochs <= ckpt["epoch"]:
        raise ValueError("--epochs must exceed the checkpoint's completed epoch")
    opt.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng"].cpu())
    if ckpt["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in ckpt["cuda_rng"]])
    return ckpt["epoch"], ckpt["history"]


def train_generative_or_cnn(args, train, val, test, metadata):
    model = {"cnn": lambda: CNN(), "vae": lambda: VAE(args.latent_dim), "ddpm": TinyUNet}[args.mode]().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    diffusion = Diffusion(args.steps, args.device) if args.mode == "ddpm" else None
    start, history = restore(model, opt, args)
    best = min([h["val_loss"] for h in history], default=float("inf"))
    output = Path(args.output)
    for epoch in range(start, args.epochs):
        model.train()
        sums, count = {}, 0
        for x, _ in loader(train, args, epoch):
            x = x.to(args.device)
            opt.zero_grad(set_to_none=True)
            if args.mode == "cnn":
                raise AssertionError("CNN uses labeled training loop below")
            if args.mode == "vae":
                logits, mu, lv = model(x)
                recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1).mean()
                kl = gaussian_kl(mu, lv)
                loss, values = recon + args.beta * kl, {"recon": recon, "kl": kl}
            else:
                t = torch.randint(args.steps, (len(x),), device=args.device)
                eps = torch.randn_like(x)
                loss = F.mse_loss(model(diffusion.noise(x, t, eps), t), eps)
                values = {}
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for key, value in {"loss": loss, **values}.items():
                sums[key] = sums.get(key, 0) + value.item() * len(x)
            count += len(x)
        model.eval()
        val_sum = baseline_sum = val_n = 0
        g = torch.Generator(device=args.device).manual_seed(args.seed + 777)
        with torch.inference_mode():
            for x, _ in loader(val, args):
                x = x.to(args.device)
                if args.mode == "vae":
                    logits, mu, lv = model(x, g)
                    value = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1).mean() + args.beta * gaussian_kl(mu, lv)
                else:
                    t = torch.randint(args.steps, (len(x),), generator=g, device=args.device)
                    eps = torch.randn(x.shape, generator=g, device=args.device)
                    value = F.mse_loss(model(diffusion.noise(x, t, eps), t), eps)
                    baseline_sum += eps.square().mean().item() * len(x)
                val_sum += value.item() * len(x)
                val_n += len(x)
        row = {"epoch": epoch + 1, "train_loss": sums["loss"] / count, "val_loss": val_sum / val_n}
        row.update({f"train_{k}": v / count for k, v in sums.items() if k != "loss"})
        if diffusion:
            row["zero_predictor_mse"] = baseline_sum / val_n
        history.append(row)
        save_checkpoint(output / "last.pt", model, opt, epoch + 1, history, args)
        if row["val_loss"] < best:
            best = row["val_loss"]
            save_checkpoint(output / "best.pt", model, opt, epoch + 1, history, args)
        print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(output / "best.pt", map_location=args.device, weights_only=True)["model"])
    with torch.inference_mode():
        x = next(iter(loader(val, args)))[0][:16].to(args.device)
        if args.mode == "vae":
            mu, _ = model.encode(x)
            recon = model.decode(mu).sigmoid()
            save_image(torch.cat([x, recon]).cpu(), output / "reconstructions.png", nrow=8)
            g = torch.Generator(device=args.device).manual_seed(args.seed + 123)
            samples = model.decode(torch.randn((16, args.latent_dim), generator=g, device=args.device)).sigmoid().cpu()
        else:
            samples = diffusion.sample(model, 16, args.seed + 123)
            noisy = []
            eps = torch.randn(x.shape, generator=torch.Generator(device=args.device).manual_seed(9), device=args.device)
            for t in (0, args.steps // 3, 2 * args.steps // 3, args.steps - 1):
                xt = diffusion.noise(x, torch.full((len(x),), t, device=args.device, dtype=torch.long), eps)
                noisy.append(((xt + 1) / 2).clamp(0, 1).cpu())
            save_image(torch.cat(noisy), output / "forward_noising.png", nrow=16)
        save_image(samples, output / "samples.png", nrow=8)
    return history, {"best_val_loss": best, "parameters": sum(p.numel() for p in model.parameters())}


def train_cnn(args, train, val, test):
    model = CNN().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start, history = restore(model, opt, args)
    best = max([h["val_accuracy"] for h in history], default=-1)
    output = Path(args.output)
    for epoch in range(start, args.epochs):
        model.train()
        total_loss = correct = count = 0
        for x, y in loader(train, args, epoch):
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        metrics = classification_eval(model, val, args)
        row = {"epoch": epoch + 1, "train_loss": total_loss / count, "val_loss": metrics["loss"],
               "train_accuracy": correct / count, "val_accuracy": metrics["accuracy"]}
        history.append(row)
        save_checkpoint(output / "last.pt", model, opt, epoch + 1, history, args)
        if metrics["accuracy"] > best:
            best = metrics["accuracy"]
            save_checkpoint(output / "best.pt", model, opt, epoch + 1, history, args)
        print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(output / "best.pt", map_location=args.device, weights_only=True)["model"])
    summary = {"best_val_accuracy": best, "val_confusion": classification_eval(model, val, args)["confusion"]}
    if test is not None:
        summary["test"] = classification_eval(model, test, args)
    return history, summary


class TinyFeatureEncoder(nn.Module):
    """Random frozen encoder for smoke tests; never labeled as DINO."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=4), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())

    def forward(self, x):
        return self.net(x)


@torch.inference_mode()
def features(encoder, data, args):
    xs, ys = [], []
    for x, y in loader(data, args):
        xs.append(encoder(x.to(args.device)).float().cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def linear_probe(args, train, val, test):
    if args.resume:
        raise ValueError("Probe is small and deterministic; rerun it instead of --resume")
    summary, main_history = {}, None
    kinds = [True, False] if args.random_baseline and args.encoder == "dinov2" else [True]
    for pretrained in kinds:
        if args.encoder == "dinov2":
            repo, kwargs = f"facebookresearch/dinov2:{args.hub_ref}", {"trust_repo": True, "skip_validation": True}
            if args.hub_dir:
                repo, kwargs = args.hub_dir, {"source": "local"}
            encoder = torch.hub.load(repo, "dinov2_vits14", pretrained=pretrained, **kwargs)
            label = "dinov2_pretrained" if pretrained else "dinov2_random"
        else:
            encoder, label = TinyFeatureEncoder(), "tiny_random_smoke"
        encoder = encoder.to(args.device).eval().requires_grad_(False)
        train_x, train_y = features(encoder, train, args)
        val_x, val_y = features(encoder, val, args)
        mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
        train_x, val_x = (train_x - mean) / std, (val_x - mean) / std
        train_features, val_features = TensorDataset(train_x, train_y), TensorDataset(val_x, val_y)
        seed_all(args.seed)
        head = nn.Linear(train_x.shape[1], 10).to(args.device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
        best, history = -1, []
        for epoch in range(args.epochs):
            head.train()
            total_loss = count = 0
            for x, y in loader(train_features, args, epoch):
                x, y = x.to(args.device), y.to(args.device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(x), y)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(y)
                count += len(y)
            metrics = classification_eval(head, val_features, args)
            row = {"epoch": epoch + 1, "train_loss": total_loss / count, "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"]}
            history.append(row)
            if metrics["accuracy"] > best:
                best = metrics["accuracy"]
                torch.save({"head": head.state_dict(), "mean": mean, "std": std,
                            "encoder": label, "hub_ref": args.hub_ref}, Path(args.output) / f"{label}_probe.pt")
            if epoch in (0, args.epochs - 1):
                print(label, json.dumps(row), flush=True)
        result = {"best_val_accuracy": best, "feature_dim": train_x.shape[1], "backbone_frozen": True}
        if test is not None:
            state = torch.load(Path(args.output) / f"{label}_probe.pt", map_location=args.device, weights_only=True)
            head.load_state_dict(state["head"])
            test_x, test_y = features(encoder, test, args)
            result["test"] = classification_eval(head, TensorDataset((test_x - mean) / std, test_y), args)
        summary[label] = result
        (Path(args.output) / f"{label}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if main_history is None:
            main_history = history
        del encoder
        if args.device == "cuda":
            torch.cuda.empty_cache()
    return main_history, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cnn", "vae", "ddpm", "probe"])
    parser.add_argument("--dataset", choices=["fashion", "synthetic"], default="fashion")
    parser.add_argument("--data-dir", default=str(ROOT / "data/datasets"))
    parser.add_argument("--output")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--encoder", choices=["dinov2", "tiny"], default="dinov2")
    parser.add_argument("--hub-ref", default=DINO_REF)
    parser.add_argument("--hub-dir", help="Existing local official DINOv2 repository for offline loading")
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="Final locked-configuration classification evaluation")
    args = parser.parse_args()
    if min(args.train_size, args.val_size, args.epochs, args.batch_size, args.latent_dim) < 1 or args.steps < 2 or args.beta < 0:
        parser.error("Sizes/epochs must be positive, steps >= 2, beta >= 0")
    if args.mode in ("vae", "ddpm") and args.evaluate_test:
        parser.error("--evaluate-test is provided for classification/probe only")
    args.device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    args.output = args.output or str(ROOT / "runs" / args.mode)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    start = time.perf_counter()
    train, val, test, metadata = data_splits(args)
    if args.mode == "cnn":
        history, result = train_cnn(args, train, val, test)
    elif args.mode == "probe":
        history, result = linear_probe(args, train, val, test)
    else:
        history, result = train_generative_or_cnn(args, train, val, test, metadata)
    metadata.update({"config": vars(args), "results": result, "history": history,
                     "seconds": time.perf_counter() - start, "python": platform.python_version(),
                     "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                     "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None})
    (Path(args.output) / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_curve(history, Path(args.output) / "curves.svg")
    print("Results:", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
