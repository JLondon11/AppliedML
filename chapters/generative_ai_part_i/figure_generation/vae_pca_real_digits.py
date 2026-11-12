"""Measured held-out VAE vs PCA reconstruction on real sklearn handwritten digits.

Run: python vae_pca_real_digits.py --out outputs/vae_pca
Results measure reconstruction, not unconditional generation quality.
"""
import argparse, csv, hashlib, json, platform, random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(64, 48)
        self.mu = nn.Linear(48, 2)
        self.logvar = nn.Linear(48, 2)
        self.dec = nn.Linear(2, 48)
        self.out = nn.Linear(48, 64)
    def encode(self, x):
        h = torch.tanh(self.enc(x))
        return self.mu(h), self.logvar(h)
    def decode(self, z):
        return torch.sigmoid(self.out(torch.tanh(self.dec(z))))
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar

def run(out: Path, seed: int = 19, epochs: int = 35):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    dataset = load_digits()
    x = dataset.data.astype("float32") / 16.0
    y = dataset.target
    ids = np.arange(len(x))
    train_val, test = train_test_split(ids, test_size=270, stratify=y, random_state=seed)
    train, val = train_test_split(train_val, test_size=270, stratify=y[train_val], random_state=seed)
    assert (len(train), len(val), len(test)) == (1257, 270, 270)
    assert len(set(train) | set(val) | set(test)) == len(x)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    xt = torch.from_numpy(x[train])
    xv = torch.from_numpy(x[val])
    generator = torch.Generator().manual_seed(seed + 10)
    history = []
    best = float("inf")
    best_weights = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(train), generator=generator).split(128):
            xb = xt[batch]
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            objective = (F.binary_cross_entropy(recon, xb, reduction="sum")
                         - 0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())) / len(xb)
            objective.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(xv)
            pred = model.decode(mu)
            val_mse = F.mse_loss(pred, xv).item()
        history.append({"epoch": epoch, "validation_reconstruction_mse": val_mse})
        if val_mse < best:
            best, best_epoch = val_mse, epoch
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test_mu, _ = model.encode(torch.from_numpy(x[test]))
        vae_recon = model.decode(test_mu).numpy()
        latent = test_mu.numpy()
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(x[train])
    pca_recon = np.clip(pca.inverse_transform(pca.transform(x[test])), 0, 1)
    vae_errors = ((vae_recon - x[test]) ** 2).mean(axis=1)
    pca_errors = ((pca_recon - x[test]) ** 2).mean(axis=1)
    improvements = pca_errors - vae_errors
    rng = np.random.default_rng(seed + 21)
    samples = rng.integers(0, len(test), size=(3000, len(test)))
    ci = np.quantile(improvements[samples].mean(axis=1), [.025, .975])
    metrics = {
        "source": "sklearn.datasets.load_digits; 8x8 scans of real handwritten digits",
        "source_samples": 1797, "train_samples": len(train),
        "validation_samples": len(val), "test_samples": len(test),
        "random_seed": seed, "epochs_max": epochs, "selected_epoch": best_epoch,
        "model": "64-48-(2 mu,2 logvar)-48-64 VAE",
        "baseline": "2-component PCA fit on training data only",
        "selection": "lowest validation reconstruction MSE for VAE; test not used for selection",
        "vae_test_reconstruction_mse_mean": float(vae_errors.mean()),
        "pca_test_reconstruction_mse_mean": float(pca_errors.mean()),
        "paired_pca_minus_vae_mse_mean": float(improvements.mean()),
        "paired_difference_bootstrap_ci95": [float(ci[0]), float(ci[1])],
        "bootstrap_draws": 3000,
        "test_index_sha256": hashlib.sha256(test.astype("<i8").tobytes()).hexdigest(),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "torch": torch.__version__},
        "limitations": ("Single seed; bootstrap quantifies test-sample variability conditional on fitted models; "
                        "VAE decoder at latent mean; reconstruction comparison only; "
                        "PCA and VAE share latent dimension but differ in capacity and objective.")
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (out / "test_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_index", "digit", "vae_mse", "pca_mse", "pca_minus_vae_mse", "latent_mu_1", "latent_mu_2"])
        for j, i in enumerate(test):
            w.writerow([int(i), int(y[i]), float(vae_errors[j]), float(pca_errors[j]),
                        float(improvements[j]), float(latent[j, 0]), float(latent[j, 1])])
    with (out / "training_history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    fig = plt.figure(figsize=(12, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    ax1.plot([h["epoch"] for h in history], [h["validation_reconstruction_mse"] for h in history], lw=2, color="#176B87")
    ax1.axvline(best_epoch, color="#A45A52", ls="--", lw=1.25)
    ax1.set(title="Validation model selection", xlabel="Epoch", ylabel="Reconstruction MSE")
    ax2.boxplot([vae_errors, pca_errors], positions=[0, 1], widths=.55, showfliers=False,
                medianprops={"color": "#A45A52", "linewidth": 2},
                boxprops={"color": "#176B87"}, whiskerprops={"color": "#176B87"},
                capprops={"color": "#176B87"})
    ax2.set(xticks=[0, 1], xticklabels=["VAE", "PCA"], title="Independent test errors", ylabel="Per-image MSE")
    scatter = ax3.scatter(latent[:, 0], latent[:, 1], c=y[test], cmap="tab10", s=13, alpha=.7, linewidths=0)
    ax3.set(title="Held-out VAE latent means", xlabel="Latent 1", ylabel="Latent 2")
    fig.colorbar(scatter, ax=ax3, label="Digit", ticks=range(10), fraction=.05)
    chosen = [int(np.flatnonzero(y[test] == digit)[0]) for digit in [0, 1, 2, 3, 4, 5, 6, 7]]
    comparison = np.concatenate([
        np.concatenate([x[test][chosen].reshape(8, 8, 8)[j] for j in range(8)], axis=1),
        np.concatenate([vae_recon[chosen].reshape(8, 8, 8)[j] for j in range(8)], axis=1),
        np.concatenate([pca_recon[chosen].reshape(8, 8, 8)[j] for j in range(8)], axis=1),
    ], axis=0)
    ax4.imshow(comparison, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax4.set(title="Independent test images: observed (top), VAE reconstruction (middle), PCA reconstruction (bottom)")
    ax4.set_yticks([3.5, 11.5, 19.5], ["Observed", "VAE", "PCA"])
    ax4.set_xticks([4 + 8*j for j in range(8)], [str(d) for d in range(8)])
    ax4.set_xlabel("Observed digit class")
    fig.savefig(out / "real_digits_vae_pca_evaluation.png", dpi=200)
    plt.close(fig)
    torch.save({"model_state_dict": model.state_dict(), "seed": seed,
                "selected_epoch": best_epoch}, out / "vae_checkpoint.pt")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/vae_pca"))
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.seed, args.epochs), indent=2))
