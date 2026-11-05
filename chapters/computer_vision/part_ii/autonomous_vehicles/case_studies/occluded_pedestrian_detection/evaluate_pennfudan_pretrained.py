#!/usr/bin/env python3
"""Real Penn-Fudan pedestrian data + pretrained torchvision Faster R-CNN inference.

Requires ORIGINAL PennFudanPed/PNGImages and PennFudanPed/PedMasks, and
official torchvision COCO weights. This is a small genuine real-image experiment,
NOT CrowdHuman, official COCO AP, or measured occlusion-stratified performance.
No metric/figure is written until real weights have loaded and inference ran.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def ground_truth(mask_path):
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        raise ValueError(f"Expected instance-ID mask, not RGB image: {mask_path}")
    objects = np.unique(mask)
    objects = objects[objects != 0]
    boxes = []
    for idx in objects:
        ys, xs = np.where(mask == idx)
        if len(xs) == 0:
            continue
        boxes.append([float(xs.min()), float(ys.min()),
                      float(xs.max() + 1), float(ys.max() + 1)])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.maximum(high - low, 0), axis=2)
    aa = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    bb = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
    return inter / np.maximum(aa[:, None] + bb[None] - inter, 1e-12)


def local_ap50(records):
    """Single class: confidence-ordered VOC-style all-point AP50."""
    total_gt = sum(len(r["gt"]) for r in records)
    if total_gt == 0:
        raise ValueError("No annotated pedestrians")
    det = sorted(
        [(float(score), j, np.asarray(box, float))
         for j, rec in enumerate(records)
         for box, score in zip(rec["pred"], rec["scores"])],
        key=lambda x: -x[0],
    )
    used = [set() for _ in records]
    tp, fp = [], []
    for score, image_idx, box in det:
        overlaps = iou_matrix([box], records[image_idx]["gt"])[0]
        if len(overlaps):
            for hit in used[image_idx]:
                overlaps[hit] = -1
        if len(overlaps) and float(overlaps.max()) >= 0.5:
            k = int(np.argmax(overlaps))
            used[image_idx].add(k)
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    t = np.cumsum(tp, dtype=float)
    f = np.cumsum(fp, dtype=float)
    recall = t / total_gt
    precision = t / np.maximum(t + f, 1)
    rr = np.r_[0., recall, 1.]
    pp = np.r_[0., precision, 0.]
    pp = np.maximum.accumulate(pp[::-1])[::-1]
    index = np.where(rr[1:] != rr[:-1])[0]
    ap = float(np.sum((rr[index + 1] - rr[index]) * pp[index + 1]))
    return {"local_ap50": ap, "ground_truth_persons": total_gt,
            "predicted_boxes": len(det), "true_positives": int(t[-1]) if len(t) else 0,
            "precision_at_last_prediction": float(precision[-1]) if len(precision) else 0.,
            "recall_at_last_prediction": float(recall[-1]) if len(recall) else 0.}


def figure(image, gt, pred, scores, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    for axes in ax:
        axes.imshow(image)
        axes.axis("off")
    for box in gt:
        x1, y1, x2, y2 = box
        ax[0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                                  fill=False, edgecolor="#198881", lw=1.7))
    for box, score in zip(pred, scores):
        x1, y1, x2, y2 = box
        ax[1].add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                                  fill=False, edgecolor="#c66b31", lw=1.7))
        ax[1].text(x1, max(y1-4, 9), f"{score:.2f}", color="#c66b31", fontsize=6)
    for j, axes in enumerate(ax):
        axes.text(.5, -.065, f"({'ab'[j]})", transform=axes.transAxes,
                  ha="center", color="black")
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.13, wspace=.045)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out.with_suffix("." + ext), dpi=190,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-images", type=int, default=12)
    p.add_argument("--score-threshold", type=float, default=.25)
    p.add_argument("--checkpoint", type=Path, help="Official downloaded torchvision weight file")
    args = p.parse_args()
    if args.n_images < 1 or not 0 <= args.score_threshold <= 1:
        raise ValueError("Invalid evaluation count or score threshold")
    root = args.dataset_root
    pictures = sorted((root / "PNGImages").glob("*.png"))
    if not pictures:
        raise FileNotFoundError("Original PennFudanPed/PNGImages missing")
    pictures = pictures[:args.n_images]
    for picture in pictures:
        label = root / "PedMasks" / (picture.stem + "_mask.png")
        if not label.is_file():
            raise FileNotFoundError(f"Missing ORIGINAL pedestrian instance mask: {label}")
    import torch
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    )
    torch.set_num_threads(2)
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    if args.checkpoint:
        if not args.checkpoint.is_file() or args.checkpoint.stat().st_size < 10_000_000:
            raise FileNotFoundError("Official pretrained model checkpoint missing")
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        checkpoint_sha = file_sha256(args.checkpoint)
    else:
        # Official weights download permitted; will throw on network/model errors.
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        import torch.hub
        checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
        checkpoint_sha = file_sha256(checkpoint_path)
    model.eval()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "Figures").mkdir(exist_ok=True)
    records = []
    prediction_rows = []
    sample_hashes = []
    for i, picture in enumerate(pictures):
        im = Image.open(picture).convert("RGB")
        tensor = weights.transforms()(im)
        gt = ground_truth(root / "PedMasks" / (picture.stem + "_mask.png"))
        with torch.inference_mode():
            pred = model([tensor])[0]
        labels = pred["labels"].cpu().numpy()
        conf = pred["scores"].cpu().numpy()
        boxes = pred["boxes"].cpu().numpy()
        keep = (labels == 1) & (conf >= args.score_threshold)
        boxes, conf = boxes[keep], conf[keep]
        records.append({"name": picture.name, "gt": gt, "pred": boxes, "scores": conf})
        for b, s in zip(boxes, conf):
            prediction_rows.append([picture.name, *map(float, b), float(s)])
        sample_hashes.append({"image": picture.name,
                              "sha256": file_sha256(picture)})
        if i < 6:
            figure(im, gt, boxes, conf, output / "Figures" / picture.stem)
    scores = local_ap50(records)
    with (output / "detections.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "x1", "y1", "x2", "y2", "score"])
        w.writerows(prediction_rows)
    with (output / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"]); w.writerows(scores.items())
    metadata = {"experiment": "PennFudanPed COCO-pretrained Faster R-CNN MobileNetV3 320",
                "source": "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip",
                "source_reference": "PyTorch torchvision object-detection tutorial",
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "weights_url": weights.url,
                "checkpoint_sha256": checkpoint_sha,
                "score_threshold": args.score_threshold,
                "evaluated_images": len(pictures), "sample_hashes": sample_hashes,
                "scope": "Local AP50, first N ordered real images; NOT official COCO AP, "
                         "not CrowdHuman, not an occlusion benchmark; no generalization claim"}
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**scores, "evaluated_images": len(pictures)}, indent=2))


if __name__ == "__main__":
    main()
