"""Numerical and lifecycle checks; no downloads, synthetic data only."""
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


labs, solutions = load("course_labs"), load("solutions")
torch.set_num_threads(2)


class MathematicalChecks(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        w = torch.tensor(3., dtype=torch.float64, requires_grad=True)
        solutions.scalar_gradient(w).backward()
        h = 1e-5
        numeric = (solutions.scalar_gradient(w.detach() + h) - solutions.scalar_gradient(w.detach() - h)) / (2*h)
        torch.testing.assert_close(w.grad, numeric)
        self.assertEqual(w.grad.item(), 4.)

    def test_nonzero_gaussian_kl_against_distribution(self):
        mu = torch.tensor([[1., -0.7], [0.2, 0.8]])
        lv = torch.tensor([[1.38629436, -.3], [.1, .2]])
        reference = kl_divergence(Normal(mu, (.5*lv).exp()), Normal(torch.zeros_like(mu), torch.ones_like(mu))).sum(1).mean()
        torch.testing.assert_close(labs.gaussian_kl(mu, lv), reference)

    def test_attention_mask_and_trainable_projection(self):
        torch.manual_seed(3)
        model = solutions.ProjectedAttention(8)
        allowed = torch.ones(4, 4, dtype=torch.bool)
        allowed[:, -1] = False
        output, weights = model(torch.randn(2, 4, 8), allowed)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))
        self.assertTrue((weights[..., -1] == 0).all())
        output.square().mean().backward()
        self.assertGreater(model.q.weight.grad.norm().item(), 0)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4, 8), torch.zeros(4, 4, dtype=torch.bool))

    def test_diffusion_conditional_moments_and_endpoint(self):
        torch.manual_seed(9)
        diffusion = labs.Diffusion(200, "cpu")
        t = torch.full((40000,), 90, dtype=torch.long)
        x = torch.full((40000, 1, 1, 1), .4)
        noisy = diffusion.noise(x, t, torch.randn_like(x)).flatten()
        self.assertAlmostEqual(noisy.mean().item(), .4*diffusion.alpha_bar[90].sqrt().item(), delta=.015)
        self.assertAlmostEqual(noisy.var().item(), 1-diffusion.alpha_bar[90].item(), delta=.02)
        self.assertEqual(diffusion.posterior_var[0].item(), 0)
        self.assertLess(diffusion.alpha_bar[-1].item(), .01)

    def test_vae_reparameterization_reaches_encoder(self):
        model = labs.VAE(4)
        logits, mu, lv = model(torch.rand(3, 1, 28, 28))
        (logits.square().mean() + labs.gaussian_kl(mu, lv)).backward()
        self.assertGreater(model.mu.weight.grad.norm().item(), 0)
        self.assertGreater(model.logvar.weight.grad.norm().item(), 0)

    def test_ddpm_backward_and_sampling_finite(self):
        model = labs.TinyUNet()
        loss = model(torch.randn(2, 1, 28, 28), torch.tensor([0, 7])).square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        samples = labs.Diffusion(8, "cpu").sample(model.eval(), 2, 7)
        self.assertEqual(tuple(samples.shape), (2, 1, 28, 28))
        self.assertTrue(torch.isfinite(samples).all())
        self.assertTrue(((samples >= 0) & (samples <= 1)).all())

    def test_sde_and_euler(self):
        exact = torch.exp(torch.tensor(-1.)).item()
        self.assertLess(abs(solutions.euler_decay(.05)-exact), abs(solutions.euler_decay(.1)-exact))
        self.assertAlmostEqual(solutions.brownian_terminal().var().item(), 1., delta=.04)


class TrainingChecks(unittest.TestCase):
    def test_incomplete_batch_loss_uses_sample_count(self):
        model = nn.Linear(3, 10)
        x, y = torch.randn(7, 3), torch.arange(7)
        args = argparse.Namespace(batch_size=4, seed=7, device="cpu")
        metrics = labs.classification_eval(model, TensorDataset(x, y), args)
        reference = nn.functional.cross_entropy(model(x), y).item()
        self.assertAlmostEqual(metrics["loss"], reference, places=6)

    def test_resume_matches_uninterrupted_cpu_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            def args(directory, epochs, resume=None):
                return argparse.Namespace(mode="cnn", dataset="synthetic", seed=7, train_size=23, val_size=11,
                    latent_dim=16, beta=1., steps=200, device="cpu", lr=.001, batch_size=8,
                    epochs=epochs, output=str(directory), resume=resume, evaluate_test=False)
            full, resumed = Path(tmp)/"full", Path(tmp)/"resumed"
            full.mkdir(); resumed.mkdir()
            train = labs.SyntheticImages(23, 7, labs.transforms.ToTensor())
            val = labs.SyntheticImages(11, 8, labs.transforms.ToTensor())
            with contextlib.redirect_stdout(io.StringIO()):
                labs.seed_all(7)
                labs.train_cnn(args(full, 2), train, val, None)
                labs.seed_all(7)
                labs.train_cnn(args(resumed, 1), train, val, None)
                labs.seed_all(19)
                labs.train_cnn(args(resumed, 2, str(resumed/"last.pt")), train, val, None)
            a = torch.load(full/"last.pt", weights_only=True)
            b = torch.load(resumed/"last.pt", weights_only=True)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            self.assertEqual(a["history"], b["history"])

    def test_frozen_backbone_only_head_gets_gradients(self):
        encoder = labs.TinyFeatureEncoder().eval().requires_grad_(False)
        with torch.inference_mode():
            x = encoder(torch.randn(4, 3, 224, 224)).clone()
        # Copy outside inference mode to obtain an ordinary tensor for head backward.
        x = x.clone()
        head = nn.Linear(x.shape[1], 10)
        nn.functional.cross_entropy(head(x), torch.tensor([0, 1, 2, 3])).backward()
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertGreater(head.weight.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
