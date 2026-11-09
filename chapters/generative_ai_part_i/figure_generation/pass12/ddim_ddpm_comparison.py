"""Matched DDPM vs deterministic DDIM experiment on the sklearn real digits dataset."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ddpm_digits_experiment import NoisePredictor, schedule, T, sample_ddpm


def ddim_timesteps(steps: int) -> list[int]:
    if not 2 <= steps <= T:
        raise ValueError("DDIM steps must be in [2, T]")
    grid = np.rint(np.linspace(0, T - 1, steps)).astype(int)
    assert len(set(grid)) == steps and grid[0] == 0 and grid[-1] == T - 1
    return grid[::-1].tolist()


@torch.no_grad()
def sample_ddim(model, initial: torch.Tensor, alpha_bar: torch.Tensor, steps: int) -> torch.Tensor:
    """DDIM eta=0 update, using alpha_bar_previous=1 at the terminal step."""
    model.eval()
    x = initial.clone()
    ts = ddim_timesteps(steps)
    for j, t in enumerate(ts):
        prev = ts[j + 1] if j + 1 < len(ts) else -1
        a = alpha_bar[t]
        ap = alpha_bar[prev] if prev >= 0 else torch.ones_like(a)
        eps = model(x, torch.full((len(x),), t, dtype=torch.long))
        x0 = (x - torch.sqrt(1 - a) * eps) / torch.sqrt(a)
        x = torch.sqrt(ap) * x0 + torch.sqrt(1 - ap) * eps
    return x


def describe(sample, training, testing):
    # Diagnostics in raw normalized [-1,1] training-image units, before display clipping.
    near = NearestNeighbors(n_neighbors=1).fit(training)
    distance = near.kneighbors(sample)[0][:, 0]
    real_distance = near.kneighbors(testing)[0][:, 0]
    return {
        "nearest_training_euclidean_mean": float(distance.mean()),
        "nearest_training_euclidean_median": float(np.median(distance)),
        "real_test_nearest_training_euclidean_mean": float(real_distance.mean()),
        "sample_pixel_outside_training_normalized_range_fraction": float(
            ((sample < -1) | (sample > 1)).mean()),
        "sample_pixel_value_mean": float(sample.mean()),
        "sample_pixel_value_sd": float(sample.std()),
    }


def run(checkpoint: Path, output: Path, samples_per_seed: int = 128,
        seeds: tuple[int, ...] = (201, 202, 203)):
    if samples_per_seed < 2 or not seeds:
        raise ValueError("Need two or more samples per seed and at least one seed")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert ckpt["steps"] == T and ckpt["seed"] == 17
    model = NoisePredictor()
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    beta, alpha, abar = schedule()
    digits = load_digits()
    normalized = digits.data.astype("float32") / 8.0 - 1.0
    indices = np.arange(len(normalized))
    tr_val, test = train_test_split(indices, test_size=270, random_state=17,
                                    stratify=digits.target)
    train, validation = train_test_split(tr_val, test_size=270, random_state=17,
                                         stratify=digits.target[tr_val])
    assert (len(train), len(validation), len(test)) == (1257, 270, 270)
    assert len(set(train) | set(validation) | set(test)) == len(indices)
    assert not (set(train)&set(validation) or set(train)&set(test)
                or set(validation)&set(test))
    configurations = [("DDPM", 100), ("DDIM", 100), ("DDIM", 50), ("DDIM", 20)]
    rows = []
    gallery = {}
    for seed in seeds:
        initial = torch.randn((samples_per_seed, 64),
                              generator=torch.Generator().manual_seed(seed))
        for method, steps in configurations:
            if method == "DDPM":
                # sample_ddpm uses the same initial seed then ancestral random draws.
                raw = sample_ddpm(model, samples_per_seed, beta, alpha, abar, seed)
            else:
                raw = sample_ddim(model, initial, abar, steps)
            sample = raw.numpy()
            assert np.isfinite(sample).all()
            metrics = describe(sample, normalized[train], normalized[test])
            rows.append({"method": method, "sampling_steps": steps, "seed": seed,
                         "sample_count": samples_per_seed, **metrics})
            if seed == seeds[0]:
                gallery[(method, steps)] = sample[:8].copy()
    with (output / "comparison_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for method, steps in configurations:
        selected = [r for r in rows if (r["method"], r["sampling_steps"]) == (method, steps)]
        summary[f"{method}-{steps}"] = {
            key: {"mean": float(np.mean([r[key] for r in selected])),
                  "sd_across_seeds": float(np.std([r[key] for r in selected], ddof=1))
                  if len(selected) > 1 else None}
            for key in ("nearest_training_euclidean_mean",
                        "sample_pixel_outside_training_normalized_range_fraction")
        }
    fig, axes = plt.subplots(5, 8, figsize=(12, 8))
    for j, digit in enumerate(normalized[test[:8]]):
        axes[0, j].imshow(((digit+1)/2).reshape(8,8), cmap="gray", vmin=0, vmax=1)
    for i, key in enumerate(configurations, 1):
        for j, sample in enumerate(gallery[key]):
            axes[i, j].imshow(np.clip((sample+1)/2,0,1).reshape(8,8),
                              cmap="gray", vmin=0, vmax=1)
    for ax in axes.flat:
        ax.axis("off")
    for i, name in enumerate(["Held-out real digits", "DDPM 100", "DDIM 100",
                               "DDIM 50", "DDIM 20"]):
        axes[i, 0].text(-2.1, 4, name, rotation=90, va="center", ha="center")
    fig.suptitle("Actual real-data DDPM/DDIM model inference", fontsize=12)
    fig.tight_layout(rect=[.03,0,1,.97])
    fig.savefig(output / "ddpm_ddim_real_digits.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(1,2,figsize=(9,3.6))
    names = list(summary)
    for ax, key, ylabel in [
        (axs[0], "nearest_training_euclidean_mean", "Nearest-training Euclidean distance"),
        (axs[1], "sample_pixel_outside_training_normalized_range_fraction",
         "Fraction of values outside [-1,1]")]:
        ax.bar(range(len(names)), [summary[n][key]["mean"] for n in names],
               yerr=[summary[n][key]["sd_across_seeds"] for n in names],
               capsize=3)
        ax.set_xticks(range(len(names)), names, rotation=20)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output / "ddpm_ddim_diagnostics.png", dpi=190)
    plt.close(fig)
    result = {
        "dataset": "sklearn.datasets.load_digits; actual 8x8 handwritten-digit images",
        "total": len(normalized), "train": len(train), "validation": len(validation),
        "test": len(test),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_training_seed": 17, "checkpoint_epoch": int(ckpt["epoch"]),
        "noise_schedule_steps": T, "sampling_seeds": list(seeds),
        "samples_per_seed": samples_per_seed, "ddim_eta": 0.,
        "ddim_uses_original_epsilon_network": True,
        "same_initial_noise_seed_per_method": True, "summary": summary,
        "limitations": "Pixel-space distance and range checks are diagnostics, not "
                       "perceptual fidelity, diversity or FID. DDIM uses the same DDPM-trained "
                       "network; this is not an independently trained DDIM model."
    }
    (output / "comparison_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("ddim_comparison"))
    p.add_argument("--samples-per-seed",type=int,default=128)
    args=p.parse_args()
    print(json.dumps(run(args.checkpoint,args.out,args.samples_per_seed),indent=2))
