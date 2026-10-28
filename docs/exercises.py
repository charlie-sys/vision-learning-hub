"""Practice before reading solutions.py. Fill each TODO and run your own checks."""
import torch


def diagonal_gaussian_kl(mu, logvar):
    """Return batch mean KL(q||N(0,I)), summing the latent dimensions."""
    raise NotImplementedError("TODO: use the Gaussian KL derivation")


def add_noise(x0, alpha_bar, epsilon):
    """alpha_bar must be broadcastable to x0; no in-place change to inputs."""
    raise NotImplementedError("TODO: combine the signal and noise terms")


def masked_attention(q, k, v, allowed):
    """Return output and weights. allowed=True means a visible key."""
    raise NotImplementedError("TODO: scale, mask, softmax over keys, aggregate")


def sample_weighted_mean(batch_means, batch_sizes):
    raise NotImplementedError("TODO: account for the final incomplete batch")


if __name__ == "__main__":
    torch.testing.assert_close(diagonal_gaussian_kl(torch.zeros(2, 3), torch.zeros(2, 3)), torch.tensor(0.0))
    print("First check passed. Add nonzero KL and non-divisible batch tests.")
