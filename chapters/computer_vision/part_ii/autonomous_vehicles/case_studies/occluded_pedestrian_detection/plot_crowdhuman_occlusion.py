"""Publication figure from observed CrowdHuman detections only.

Reads per_person_results.csv written by evaluate_crowdhuman_frcnn.py.
Scene-cluster resampling keeps persons from the same photograph together.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BINS = ("low", "medium", "heavy")
LABELS = ("Low (0–20%)", "Moderate (20–50%)", "Heavy (>=50%)")


def summarize(path: Path, resamples=5000, seed=20261111):
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples required")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No measured pedestrian records: refusing empty figure")
    scenes = {}
    for row in rows:
        group = row["occlusion_bin"]
        hit = int(row["matched_IoU50"])
        if group not in BINS or hit not in (0, 1):
            raise ValueError("Invalid original detection record")
        scenes.setdefault(row["image"], []).append((BINS.index(group), hit))
    arr = np.zeros((len(scenes), 3, 2), dtype=np.int64)
    for index, scene in enumerate(sorted(scenes)):
        for bin_index, hit in scenes[scene]:
            arr[index, bin_index, 0] += hit
            arr[index, bin_index, 1] += 1
    matched = arr[:, :, 0].sum(axis=0)
    total = arr[:, :, 1].sum(axis=0)
    if np.any(total == 0):
        raise ValueError("Each bin must contain observed original people")
    mean = matched / total
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, 3))
    for j in range(resamples):
        sample = arr[rng.integers(0, len(scenes), size=len(scenes))].sum(axis=0)
        draws[j] = np.divide(sample[:, 0], sample[:, 1],
                             out=np.full(3, np.nan), where=sample[:, 1] > 0)
    intervals = np.array([np.nanquantile(draws[:, i], [.025, .975])
                          for i in range(3)])
    return {
        "source_image_count": len(scenes), "original_person_count": len(rows),
        "matched": matched.tolist(), "total": total.tolist(),
        "recall": mean.tolist(), "ci_lower": intervals[:, 0].tolist(),
        "ci_upper": intervals[:, 1].tolist(),
        "resamples": resamples, "seed": seed,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "interval_definition": "Exploratory original-image-cluster percentile bootstrap",
        "metric_definition": "Local full-body person recall at IoU>=0.5; not official CrowdHuman AP/MR/JI"
    }


def render(result, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    x = np.arange(3)
    y = np.asarray(result["recall"])
    lower = np.asarray(result["ci_lower"])
    upper = np.asarray(result["ci_upper"])
    fig, ax = plt.subplots(figsize=(7.7, 4.5), constrained_layout=True)
    ax.vlines(x, lower, upper, linewidth=3, color="#00858B", zorder=2)
    ax.scatter(x, y, s=85, color="#123B54", zorder=3)
    for i, count in enumerate(result["total"]):
        ax.text(i, .04, f"n={count:,}", ha="center", fontsize=10, color="#30485A")
    ax.set(ylim=(0, 1.02), xlim=(-.4, 2.4),
           ylabel="Observed recall at IoU >= 0.50",
           title="CrowdHuman: measured detection recall by occlusion")
    ax.set_xticks(x, LABELS)
    ax.grid(axis="y", color="#E4E9ED", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(output / "crowdhuman_occlusion_recall.pdf", bbox_inches="tight")
    fig.savefig(output / "crowdhuman_occlusion_recall.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (output / "occlusion_uncertainty.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261111)
    args = p.parse_args()
    render(summarize(args.input_csv, args.resamples, args.seed), args.output)


if __name__ == "__main__":
    main()
