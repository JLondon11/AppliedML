#!/usr/bin/env python3
"""Run OFFICIAL pretrained COCO YOLO11n on genuine VisDrone aerial imagery.

This is a *cross-domain* zero-finetuning baseline, NOT a VisDrone-trained YOLO
or an agricultural obstacle-detection experiment. Only unambiguous class pairs
are evaluated. Ignore regions and unmatched VisDrone-specific classes are not
used in this local AP50; results are NOT official VisDrone challenge AP.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

# (VisDrone original 1-based class ID, COCO 0-based class ID, class name)
MAPPING = (
    (1, 0, "pedestrian"),
    (3, 1, "bicycle"),
    (4, 2, "car"),
    (6, 7, "truck"),
    (9, 5, "bus"),
    (10, 3, "motor"),
)
VD_TO_COCO = {vd: coco for vd, coco, _ in MAPPING}
COCO_TO_VD = {coco: vd for vd, coco, _ in MAPPING}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_original_gt(txt):
    """VisDrone x,y,w,h,score,category,truncation,occlusion.

    Return only the six mapped classes, excluding ignored objects. 'people'
    (seated/other posture) intentionally is NOT merged into COCO person.
    """
    gt = []
    for lineno, line in enumerate(txt.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 8:
            raise ValueError(f"Malformed original annotation {txt}:{lineno}")
        x, y, w, h, flag, cat = map(float, parts[:6])
        if w <= 0 or h <= 0 or not np.isfinite([x, y, w, h]).all():
            continue
        cat = int(cat)
        if int(flag) == 0 or cat not in VD_TO_COCO:
            continue
        gt.append((x, y, x + w, y + h, cat))
    return np.asarray(gt, dtype=float).reshape(-1, 5)


def iou(box, others):
    others = np.asarray(others, float).reshape(-1, 4)
    if not len(others):
        return np.zeros(0, dtype=float)
    box = np.asarray(box, float)
    lo = np.maximum(box[:2], others[:, :2])
    hi = np.minimum(box[2:], others[:, 2:])
    intersection = np.prod(np.maximum(hi - lo, 0), axis=1)
    area_a = np.prod(np.maximum(box[2:] - box[:2], 0))
    area_b = np.prod(np.maximum(others[:, 2:] - others[:, :2], 0), axis=1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def ap50(records, class_id):
    ground_truths = {
        r["name"]: r["gt"][r["gt"][:, 4] == class_id, :4]
        for r in records
    }
    n_gt = sum(len(v) for v in ground_truths.values())
    detections = sorted(
        [(float(p[4]), r["name"], p[:4])
         for r in records for p in r["pred"] if int(p[5]) == class_id],
        key=lambda item: -item[0],
    )
    if not n_gt:
        return None, 0, len(detections), 0, 0
    matched = {k: set() for k in ground_truths}
    tp, fp = [], []
    for score, name, box in detections:
        overlaps = iou(box, ground_truths[name])
        if len(overlaps):
            for idx in matched[name]:
                overlaps[idx] = -1.
        if len(overlaps) and float(overlaps.max()) >= .5:
            idx = int(overlaps.argmax())
            matched[name].add(idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    ct = np.cumsum(tp, dtype=float)
    cf = np.cumsum(fp, dtype=float)
    recall = ct / n_gt
    precision = ct / np.maximum(ct + cf, 1)
    rc = np.r_[0, recall, 1]
    pc = np.r_[0, precision, 0]
    pc = np.maximum.accumulate(pc[::-1])[::-1]
    change = np.where(rc[1:] != rc[:-1])[0]
    ap = float(np.sum((rc[change + 1] - rc[change]) * pc[change + 1]))
    return ap, n_gt, len(detections), int(ct[-1]) if len(ct) else 0, int(cf[-1]) if len(cf) else 0


def render(bgr, gt, pred, path):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.15), facecolor="white")
    for ax in axes:
        ax.imshow(rgb)
        ax.axis("off")
    for x1, y1, x2, y2, cls in gt:
        axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                           fill=False, edgecolor="#087e79", lw=1.1))
    for x1, y1, x2, y2, confidence, cls in pred:
        axes[1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                           fill=False, edgecolor="#c06b3b", lw=1.1))
        axes[1].text(x1, max(y1-2, 8), f"{confidence:.2f}",
                     color="#c06b3b", fontsize=5)
    for i, ax in enumerate(axes):
        ax.text(.5, -.06, "(" + "ab"[i] + ")", transform=ax.transAxes,
                ha="center", color="black", fontsize=10)
    fig.subplots_adjust(left=.012, right=.988, top=.98, bottom=.13, wspace=.025)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix("." + ext), dpi=190,
                    bbox_inches="tight", pad_inches=.07, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint", default="yolo11n.pt")
    p.add_argument("--max-images", type=int, default=24)
    p.add_argument("--max-figures", type=int, default=5)
    p.add_argument("--conf", type=float, default=.01)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    images = sorted(args.images.glob("*.jpg"))[:args.max_images]
    if len(images) != args.max_images or args.max_images <= 0:
        raise ValueError("Insufficient original VisDrone image files")
    for path in images:
        if not (args.annotations / f"{path.stem}.txt").is_file():
            raise FileNotFoundError(f"Missing original annotation for {path.name}")
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version
    torch.set_num_threads(2)
    model = YOLO(args.checkpoint)
    if {0, 1, 2, 3, 5, 7}.difference(set(model.names)):
        raise ValueError("Expected canonical COCO YOLO categories, not a VisDrone model")
    if str(model.names[0]).lower() != "person" or str(model.names[2]).lower() != "car":
        raise ValueError("Expected official COCO class labels")
    weights_path = Path(model.ckpt_path or args.checkpoint)
    if not weights_path.is_file():
        raise FileNotFoundError("Downloaded original YOLO checkpoint missing")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "Figures").mkdir(exist_ok=True)
    records, prediction_rows, source_hashes = [], [], []
    for j, imgpath in enumerate(images):
        bgr = cv2.imread(str(imgpath))
        if bgr is None:
            raise ValueError("Cannot load VisDrone original image " + imgpath.name)
        gtpath = args.annotations / f"{imgpath.stem}.txt"
        gt = parse_original_gt(gtpath)
        res = model.predict(bgr, imgsz=args.imgsz, conf=args.conf,
                            device="cpu", verbose=False)[0]
        pred = []
        if res.boxes is not None:
            for box, score, cls in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
            ):
                cls = int(cls)
                if cls in COCO_TO_VD:
                    pred.append([*box, float(score), COCO_TO_VD[cls]])
                    prediction_rows.append([imgpath.name, COCO_TO_VD[cls],
                                            float(score), *map(float, box)])
        pred = np.asarray(pred, dtype=float).reshape(-1, 6)
        records.append(dict(name=imgpath.name, gt=gt, pred=pred))
        source_hashes.append(dict(file=imgpath.name, sha256=sha256(imgpath),
                                  labels_sha256=sha256(gtpath)))
        if j < args.max_figures:
            # Display a stricter threshold for legible publication panels only.
            # AP50 evaluation and exported predictions still use --conf (default 0.01).
            display_pred = pred[pred[:, 4] >= .25]
            render(bgr, gt, display_pred, args.output / "Figures" / imgpath.stem)
    per_class = []
    for vd, coco, name in MAPPING:
        ap, n_gt, n_det, tp, fp = ap50(records, vd)
        per_class.append([name, vd, coco, ap, n_gt, n_det, tp, fp])
    with (args.output / "per_class_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "visdrone_category", "coco_category", "local_ap50",
                         "original_gt_count", "detections", "true_positives", "false_positives"])
        writer.writerows(per_class)
    with (args.output / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "visdrone_category", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(prediction_rows)
    valid = [row[3] for row in per_class if row[3] is not None]
    overall = dict(n_images=len(images), n_original_gt=sum(row[4] for row in per_class),
                   n_detections=len(prediction_rows),
                   local_macro_AP50=float(np.mean(valid)) if valid else None,
                   local_micro_recall_at_cutoff=(
                     sum(row[6] for row in per_class) / max(1, sum(row[4] for row in per_class))),
                   class_count_evaluated=len(valid))
    with (args.output / "summary_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["metric", "value"])
        writer.writerows(overall.items())
    manifest = dict(
        dataset="Original VisDrone2019-DET-val images/annotations",
        dataset_url="https://github.com/VisDrone/VisDrone-Dataset",
        model="Official Ultralytics YOLO11n pretrained on COCO (no VisDrone finetuning)",
        checkpoint=str(weights_path),
        checkpoint_sha256=sha256(weights_path),
        ultralytics_version=ultralytics_version,
        torch_version=torch.__version__,
        imgsz=args.imgsz, confidence_cutoff=args.conf, figure_confidence_cutoff=0.25,
        mapping=[dict(visdrone=vd, coco=coco, name=name) for vd,coco,name in MAPPING],
        source_files=source_hashes,
        scope="Small sorted real validation subset; local AP50 computed without official ignored-region matching. NOT official VisDrone challenge AP; NOT agricultural field data; no tracking performance.",
    )
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
