"""Figure 16: FID, Precision, Recall, and Coverage under a common published evaluation.

Published CIFAR-10 values reproduced from Table 1 of:
Mohammad Jalali, Cheuk Ting Li, Farzan Farnia,
"An Information-Theoretic Evaluation of Generative Models in Learning Multi-modal Distributions,"
NeurIPS 2023.

This script preserves the raw published values. Color encodes only within-column
relative performance; FID is inverted for the color scale because lower is better.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "figure16_cifar10_published.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA)

perf = pd.DataFrame(index=data.index)
perf["FID"] = 1 - (data["FID"] - data["FID"].min()) / (data["FID"].max() - data["FID"].min())
for c in ["Precision", "Recall", "Coverage"]:
    perf[c] = (data[c] - data[c].min()) / (data[c].max() - data[c].min())

cmap = LinearSegmentedColormap.from_list(
    "premium_blue", ["#F7F9FB", "#D8E3EA", "#8AA6B6", "#3E6478", "#173A52"]
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, ax = plt.subplots(figsize=(7.4, 4.7), dpi=240)
im = ax.imshow(perf[["FID", "Precision", "Recall", "Coverage"]].to_numpy(),
               cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(4), ["FID ↓", "Precision ↑", "Recall ↑", "Coverage ↑"])
ax.set_yticks(range(len(data)), data["Model"])

for i, row in data.iterrows():
    vals = [row["FID"], row["Precision"], row["Recall"], row["Coverage"]]
    for j, val in enumerate(vals):
        text_color = "white" if perf.iloc[i, j] > 0.62 else "#17212A"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color=text_color, fontsize=9)

ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-.5, len(data), 1), minor=True)
ax.grid(which="minor", linewidth=0.45, alpha=0.35)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
cbar.set_label("Within-column relative performance", rotation=90)
cbar.set_ticks([0, 0.5, 1.0])
cbar.ax.set_yticklabels(["lower", "mid", "higher"])

fig.tight_layout()
fig.savefig(OUT / "figure16_metrics.png", bbox_inches="tight")
fig.savefig(OUT / "figure16_metrics.pdf", bbox_inches="tight")
