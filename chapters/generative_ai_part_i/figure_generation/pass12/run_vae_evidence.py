"""Reproduce the real-digits VAE evidence underlying Chapter 9 figures 10, 11 and 14.

Run: python run_vae_evidence.py --out outputs/vae_audit
The original chapter uses validation data (not a held-out test set).
"""
import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(64, 48)
        self.mu = nn.Linear(48, 2)
        self.lv = nn.Linear(48, 2)
        self.d1 = nn.Linear(2, 48)
        self.out = nn.Linear(48, 64)

    def encode(self, x):
        h = torch.tanh(self.e1(x))
        return self.mu(h), self.lv(h)

    def decode(self, z):
        return torch.sigmoid(self.out(torch.tanh(self.d1(z))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


def loss_terms(model, x, deterministic=False):
    if deterministic:
        mu, logvar = model.encode(x)
        reconstruction = model.decode(mu)
    else:
        reconstruction, mu, logvar = model(x)
    bce = F.binary_cross_entropy(reconstruction, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())
    return bce, kl


def run(out: Path, epochs: int = 60, seed: int = 13):
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    digits = load_digits()
    X = digits.data.astype("float32") / 16.0
    y = digits.target.astype("int64")
    indices = np.arange(len(X))
    tr_idx, va_idx = train_test_split(indices, test_size=0.25, random_state=seed, stratify=y)
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    assert len(tr_idx) == 1347 and len(va_idx) == 450
    assert not set(tr_idx).intersection(va_idx)
    assert X.min() >= 0 and X.max() <= 1 and len(np.unique(y)) == 10
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr)), batch_size=128, shuffle=True)
    validation = DataLoader(TensorDataset(torch.from_numpy(Xva)), batch_size=256, shuffle=False)
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_bce = train_kl = 0.0
        for (xb,) in loader:
            optimizer.zero_grad()
            bce, kl = loss_terms(model, xb)
            (bce + kl).backward()
            optimizer.step()
            train_bce += bce.item()
            train_kl += kl.item()
        model.eval()
        val_bce = val_kl = 0.0
        with torch.no_grad():
            for (xb,) in validation:
                bce, kl = loss_terms(model, xb)
                val_bce += bce.item()
                val_kl += kl.item()
        history.append({"epoch": epoch, "train_bce": train_bce / len(Xtr),
                        "train_kl": train_kl / len(Xtr),
                        "train_elbo": (train_bce + train_kl) / len(Xtr),
                        "val_bce": val_bce / len(Xva),
                        "val_kl": val_kl / len(Xva),
                        "val_elbo": (val_bce + val_kl) / len(Xva)})
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(torch.tensor(Xva))
        Z = mu.numpy()
        reconstruction = model.decode(mu).numpy()
        bce_det, kl_det = loss_terms(model, torch.tensor(Xva), deterministic=True)
    neighbors = NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z, return_distance=False)
    # Exclude the query itself, including if tied embeddings change neighbor ordering.
    others = np.array([row[row != i][:10] for i, row in enumerate(neighbors)], dtype=int)
    assert others.shape == (len(Xva), 10)
    purity = (yva[others] == yva[:, None]).mean(axis=1)
    assert np.isfinite(purity).all()
    with (out / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (out / "validation_embeddings.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_index", "true_digit", "latent_mu_1", "latent_mu_2", "knn_label_purity"])
        writer.writerows(zip(va_idx, yva, Z[:, 0], Z[:, 1], purity))
    metrics = {
        "evidence_type": "measured_validation_not_test",
        "source": "sklearn.datasets.load_digits",
        "dataset_samples": len(X), "train_samples": len(Xtr), "validation_samples": len(Xva),
        "test_samples": 0, "features": 64, "classes": 10, "seed": seed, "epochs": epochs,
        "batch_size": 128, "optimizer": "Adam", "learning_rate": 0.002,
        "final_train_elbo": history[-1]["train_elbo"],
        "final_validation_elbo_stochastic": history[-1]["val_elbo"],
        "final_validation_bce_deterministic": bce_det.item() / len(Xva),
        "final_validation_kl": kl_det.item() / len(Xva),
        "validation_latent_10nn_purity_mean": float(purity.mean()),
        "validation_latent_10nn_purity_sd": float(purity.std(ddof=1)),
        "sample_index_hash_sha256": hashlib.sha256(va_idx.astype("<i8").tobytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "sklearn": sklearn.__version__, "torch": torch.__version__,
        "figure_numbers": [10, 11, 14],
        "limitations": "A validation split only; no holdout test set or independent benchmark comparison."
    }
    assert np.isfinite([v for v in metrics.values() if isinstance(v, float)]).all()
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].plot([x["epoch"] for x in history], [x["train_elbo"] for x in history], label="train")
    axs[0].plot([x["epoch"] for x in history], [x["val_elbo"] for x in history], label="validation")
    axs[0].set(xlabel="Epoch", ylabel="Negative ELBO / sample")
    axs[0].legend()
    axs[1].scatter(Z[:, 0], Z[:, 1], c=yva, s=10, cmap="tab10")
    axs[1].set(xlabel="Latent mean 1", ylabel="Latent mean 2")
    fig.tight_layout()
    fig.savefig(out / "vae_validation.png", dpi=180)
    plt.close(fig)
    with torch.no_grad():
        i1 = int(np.flatnonzero(yva == 1)[0])
        i7 = int(np.flatnonzero(yva == 7)[0])
        anchor = torch.tensor(Z[[i1, i7]])
        a = torch.linspace(0, 1, 8).view(-1, 1)
        interpolation = model.decode((1-a)*anchor[0] + a*anchor[1]).reshape(-1, 8, 8).numpy()
    fig, axs = plt.subplots(1, 8, figsize=(9, 1.4))
    for ax, im in zip(axs, interpolation):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "vae_interpolation.png", dpi=180)
    plt.close(fig)
    torch.save({"state_dict": model.state_dict(), "seed": seed}, out / "vae_model.pt")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/vae_audit"))
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()
    if args.epochs < 1:
        p.error("--epochs must be positive")
    print(json.dumps(run(args.out, args.epochs), indent=2))
