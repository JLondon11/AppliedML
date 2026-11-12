"""Plot audited published-in-repository YOLO11n detections on genuine VisDrone photos.

This replot does not re-execute detector inference. It aborts if the two
existing CSV source summaries do not reconcile arithmetically.
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


def load(per_class_path, summary_path):
    with per_class_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with summary_path.open(newline="", encoding="utf-8") as f:
        summary = {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}
    if not rows:
        raise ValueError("No measured rows")
    valid = [r for r in rows if int(r["original_gt_count"]) > 0]
    if len(valid) != int(summary["class_count_evaluated"]):
        raise ValueError("Class count mismatch")
    gt = sum(int(r["original_gt_count"]) for r in rows)
    det = sum(int(r["detections"]) for r in rows)
    if (gt, det) != (int(summary["n_original_gt"]), int(summary["n_detections"])):
        raise ValueError("Dataset count mismatch")
    ap = np.array([float(r["local_ap50"]) for r in valid])
    if not np.isfinite(ap).all() or (ap < 0).any() or (ap > 1).any():
        raise ValueError("Invalid original class AP")
    macro = float(ap.mean())
    recall = sum(int(r["true_positives"]) for r in valid) / gt
    if abs(macro-summary["local_macro_AP50"]) > 1e-10:
        raise ValueError("Macro AP reconciliation failure")
    if abs(recall-summary["local_micro_recall_at_cutoff"]) > 1e-10:
        raise ValueError("Micro recall reconciliation failure")
    for r in rows:
        n = int(r["original_gt_count"])
        d = int(r["detections"])
        tp = int(r["true_positives"])
        fp = int(r["false_positives"])
        if n and d != tp+fp:
            raise ValueError("TP/FP/detections do not reconcile")
    return rows, valid, summary, ap, macro, recall


def render(per_class_path, summary_path, output):
    rows, valid, summary, ap, macro, recall = load(per_class_path, summary_path)
    output.mkdir(parents=True, exist_ok=True)
    y = np.arange(len(valid))
    fig, (ax, axn) = plt.subplots(
        1, 2, figsize=(10, 5.6), sharey=True,
        gridspec_kw={"width_ratios": [3.3, 1.25], "wspace": .04})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=.83, bottom=.26, left=.15, right=.97)
    ax.hlines(y, 0, ap, color="#A4C6CD", lw=2.4)
    ax.scatter(ap, y, color="#13546C", edgecolors="white", lw=1.1, s=85, zorder=3)
    for yi, v in zip(y, ap):
        ax.text(v+.012, yi, f"{v:.3f}", va="center", fontsize=9, color="#173C50")
    ax.axvline(macro, color="#A66147", linestyle=(0, (4, 4)), lw=1.7,
               label=f"Macro AP50 = {macro:.3f}")
    ax.set(xlim=(0,.60), ylim=(-.5,len(valid)-.5), xlabel="Local AP at IoU = 0.50")
    ax.set_yticks(y, [r["class"].capitalize() for r in valid])
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0,.61,.1))
    ax.grid(axis="x", color="#E1E7EB", lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    axn.set_xlim(0,1)
    axn.set_xticks([])
    axn.tick_params(axis="y", length=0,labelleft=False)
    for sp in axn.spines.values():
        sp.set_visible(False)
    axn.text(.08, 1.045, "Original GT", transform=axn.transAxes,
             fontsize=9, weight="bold", color="#173C50")
    axn.text(.48, 1.045, "TP / Detections", transform=axn.transAxes,
             fontsize=9, weight="bold", color="#173C50")
    for yi,r in zip(y,valid):
        axn.text(.10,yi,f'{int(r["original_gt_count"]):,}',va="center",fontsize=9,color="#425A69")
        axn.text(.50,yi,f'{int(r["true_positives"])} / {int(r["detections"]):,}',
                 va="center",fontsize=9,color="#425A69")
    fig.suptitle("Measured YOLO11n single-frame transfer to original VisDrone images",
                 x=.53,y=.95,fontsize=14,color="#173C50")
    fig.text(.15,.10,f'24 real validation frames | 972 original GT objects | Recall at recorded cutoff: {recall:.3f}',
             fontsize=9,color="#425A69")
    fig.text(.15,.057,"Bus: no ground-truth instances; AP undefined and excluded from macro.",
             fontsize=9,color="#425A69")
    fig.text(.15,.025,
             "Local cross-domain detection results; not official VisDrone AP, drone tracking, planning or control.",
             fontsize=8.5,color="#516572")
    for extension in ("png","pdf","svg"):
        fig.savefig(output/f"visdrone_yolo_transfer.{extension}",
                    dpi=240,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    evidence = {
        "source":"Reported actual inference on 24 original VisDrone validation frames",
        "n_images":int(summary["n_images"]),"n_gt":int(summary["n_original_gt"]),
        "n_detections":int(summary["n_detections"]),
        "n_evaluated_classes":len(valid),"recomputed_macro_AP50":macro,
        "recomputed_micro_recall":recall,
        "undefined_AP_excluded":[r["class"] for r in rows if int(r["original_gt_count"])==0],
        "source_sha256":{
            "per_class_metrics.csv":hashlib.sha256(per_class_path.read_bytes()).hexdigest(),
            "summary_metrics.csv":hashlib.sha256(summary_path.read_bytes()).hexdigest()},
        "limit":"Replot of existing measured local single-frame transfer, not a new experiment or an official leaderboard result"
    }
    (output/"source_metrics_verified.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--per-class", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    render(a.per_class,a.summary,a.output)


if __name__=="__main__":
    main()
