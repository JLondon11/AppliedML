"""Figure 15: Generative Quality-Diversity Tradeoff.

Reproducible controlled distribution experiment. This script does NOT claim
published model-family benchmark performance. It measures empirical support
precision/recall for controlled generated distributions relative to a fixed
four-mode target distribution and saves both numerical results and artwork.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

SEED = 15015
N = 5000
BOOTSTRAPS = 250
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

centers = np.array([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]])

def sample_mix(centers_used, sigma, n=N):
    idx = rng.integers(0, len(centers_used), n)
    return centers_used[idx] + rng.normal(0, sigma, size=(n, 2))

real = sample_mix(centers, 0.48)
conditions = {
    "Mode-seeking": sample_mix(centers[:2], 0.30),
    "Balanced": sample_mix(centers, 0.50),
    "Over-dispersed": sample_mix(centers, 0.82),
    "Shifted": sample_mix(centers + np.array([0.45, -0.25]), 0.48),
}

real_tree = cKDTree(real)
d_rr, _ = real_tree.query(real, k=2)
tau = np.quantile(d_rr[:, 1], 0.95)

def support_pr(generated):
    generated_tree = cKDTree(generated)
    d_g_to_r, _ = real_tree.query(generated, k=1)
    d_r_to_g, _ = generated_tree.query(real, k=1)
    return np.mean(d_g_to_r <= tau), np.mean(d_r_to_g <= tau)

rows = []
for name, generated in conditions.items():
    precision, recall = support_pr(generated)
    bootstrap = []
    for _ in range(BOOTSTRAPS):
        resample = generated[rng.integers(0, len(generated), len(generated))]
        bootstrap.append(support_pr(resample))
    bootstrap = np.asarray(bootstrap)
    rows.append({
        "condition": name,
        "precision": precision,
        "recall": recall,
        "precision_bootstrap_sd": bootstrap[:, 0].std(ddof=1),
        "recall_bootstrap_sd": bootstrap[:, 1].std(ddof=1),
    })

results = pd.DataFrame(rows)
results.to_csv(OUT / "figure15_results.csv", index=False)

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.labelsize": 11, "legend.fontsize": 9})
fig, ax = plt.subplots(figsize=(7.1, 5.0), dpi=220)
markers = ["o", "s", "^", "D"]
for row, marker in zip(rows, markers):
    ax.errorbar(row["recall"], row["precision"],
                xerr=row["recall_bootstrap_sd"],
                yerr=row["precision_bootstrap_sd"],
                fmt=marker, ms=7, capsize=3, linewidth=1.1,
                label=row["condition"])
ax.set_xlabel("Empirical support recall")
ax.set_ylabel("Empirical support precision")
ax.set_xlim(0, 1.03)
ax.set_ylim(0, 1.03)
ax.grid(True, linewidth=0.5, alpha=0.28)
ax.legend(frameon=False, loc="lower left")
ax.text(1.0, 0.02, f"N={N} per distribution; bootstrap B={BOOTSTRAPS}",
        ha="right", va="bottom", transform=ax.transAxes, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "figure15_quality_diversity.png", bbox_inches="tight")
fig.savefig(OUT / "figure15_quality_diversity.pdf", bbox_inches="tight")
print(results.to_string(index=False))
print(f"support threshold tau={tau:.6f}")
