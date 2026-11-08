#!/usr/bin/env python3
"""Plot ALL genuine SAM-versus-GrabCut paired detections, not a six-item pilot.

Source CSV: per_object.csv downloaded from successful original COCO SAM execution
https://github.com/JLondon11/AppliedML/actions/runs/42300131086
The plot uses only original measured objects (78 valid pairs out of 120).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(path, output):
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    paired = [r for r in rows if r["grabcut_iou"] != ""]
    if len(rows) != 120 or len(paired) != 78:
        raise ValueError("Expected 120 original SAM rows and 78 valid GrabCut pairs")
    sam = np.array([float(r["sam_iou"]) for r in paired])
    grabcut = np.array([float(r["grabcut_iou"]) for r in paired])
    if not np.all(np.isfinite(sam)) or not np.all(np.isfinite(grabcut)):
        raise ValueError("Measured masks contain invalid score")
    delta = sam - grabcut
    if abs(np.mean(delta) - 0.24541314269301318) > 1e-9:
        raise AssertionError("Mismatch with archived measured pair differences")
    sam_wins = int((delta > 0).sum())
    grabcut_wins = int((delta < 0).sum())
    if (sam_wins, grabcut_wins) != (66, 12):
        raise AssertionError("Observed paired winners mismatch original evidence")
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.25),
                            gridspec_kw={"width_ratios": [1.05, 1]},
                            layout="constrained")
    axes[0].scatter(grabcut[delta > 0], sam[delta > 0], s=27,
                    alpha=.72, c="#267E91", marker="o", edgecolors="white",
                    linewidths=.3, label=f"SAM higher ({sam_wins})")
    axes[0].scatter(grabcut[delta < 0], sam[delta < 0], s=32,
                    alpha=.85, c="#B47654", marker="D", edgecolors="white",
                    linewidths=.3, label=f"GrabCut higher ({grabcut_wins})")
    axes[0].plot([0, 1], [0, 1], color="#757575", linestyle="--", lw=1)
    axes[0].set(xlim=(-.035, 1.035), ylim=(-.035, 1.035),
                xlabel="GrabCut mask IoU", ylabel="SAM ViT-B mask IoU")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[1].hist(delta, bins=np.linspace(-.65, 1.02, 19),
                 color="#267E91", edgecolor="white", linewidth=.7)
    axes[1].axvline(0, lw=1, color="#757575", ls="--")
    axes[1].axvline(np.mean(delta), lw=1.7, color="#B47654",
                    label=f"Mean difference = {np.mean(delta):.3f}")
    axes[1].set(xlabel="Paired IoU difference (SAM − GrabCut)",
                ylabel="Evaluated objects")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    for ax in axes:
        ax.grid(alpha=.12)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + ext), dpi=210,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    result = {"evaluated_sam_objects": len(rows), "paired_objects": len(paired),
              "sam_higher": sam_wins, "grabcut_higher": grabcut_wins,
              "paired_mean_sam": float(np.mean(sam)),
              "paired_mean_grabcut": float(np.mean(grabcut)),
              "paired_mean_difference": float(np.mean(delta)),
              "paired_median_difference": float(np.median(delta))}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.csv, args.output)
