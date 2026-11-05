"""CPU-only, genuine SAM ViT-B / COCO 2017 GT-box prompted segmentation.
All metrics are computed from model predictions on official COCO validation images.
This is NOT official COCO AP. Results are exploratory subset measurements.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor

ANNOTATIONS = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_ROOT = "https://images.cocodataset.org/val2017/"
CHECKPOINT = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def download(url, path):
    path = Path(path)
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(url, timeout=240) as source, open(temp, "wb") as target:
        while True:
            block = source.read(1048576)
            if not block:
                break
            target.write(block)
    temp.replace(path)
    return path


def score(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    intersection = int(np.count_nonzero(pred & truth))
    union = int(np.count_nonzero(pred | truth))
    total = int(pred.sum() + truth.sum())
    return (intersection / union if union else 1.0,
            2 * intersection / total if total else 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--images", type=int, default=8)
    p.add_argument("--max-objects", type=int, default=24)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--min-area", type=int, default=3000)
    args = p.parse_args()
    if args.images < 1 or args.max_objects < 1:
        p.error("--images and --max-objects must be positive")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    args.out.mkdir(parents=True, exist_ok=True)
    archive = download(ANNOTATIONS, args.work / "annotations_trainval2017.zip")
    with zipfile.ZipFile(archive) as z:
        raw = json.load(z.open("annotations/instances_val2017.json"))
    annotation_path = args.work / "instances_val2017.json"
    annotation_path.write_text(json.dumps(raw))
    eligible = sorted({int(a["image_id"]) for a in raw["annotations"]
                       if not a.get("iscrowd") and a.get("area", 0) >= args.min_area})
    if len(eligible) < args.images:
        raise RuntimeError("Not enough eligible COCO images")
    selected = random.Random(args.seed).sample(eligible, args.images)
    info = {int(x["id"]): x for x in raw["images"]}
    checkpoint = download(CHECKPOINT, args.work / "sam_vit_b_01ec64.pth")
    if checkpoint.stat().st_size < 10000000:
        raise RuntimeError("Checkpoint is missing or truncated")
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to("cpu")
    model.eval()
    predictor = SamPredictor(model)
    coco = COCO(str(annotation_path))
    rows, images = [], {}
    for image_id in selected:
        item = info[image_id]
        path = download(IMAGE_ROOT + item["file_name"], args.work / "val2017" / item["file_name"])
        images[item["file_name"]] = sha256(path)
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError("Cannot decode source image " + str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))
        predictor.set_image(rgb)
        for a in anns:
            if a.get("area", 0) < args.min_area:
                continue
            x, y, bw, bh = map(float, a["bbox"])
            box = np.array([max(0., x), max(0., y), min(w, x + bw), min(h, y + bh)], np.float32)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            gt = coco.annToMask(a).astype(bool)
            if not np.any(gt):
                continue
            with torch.inference_mode():
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
            sam_iou, sam_dice = score(masks[0], gt)
            # Failed baseline attempts remain explicit missing values, never zero.
            grabcut_iou = grabcut_dice = ""
            bx = np.round(box).astype(int)
            x1, y1, x2, y2 = map(int, bx)
            if x1 > 0 and y1 > 0 and x2 < w - 1 and y2 < h - 1 and x2 - x1 > 1 and y2 - y1 > 1:
                try:
                    gc_mask = np.zeros((h, w), np.uint8)
                    bg = np.zeros((1, 65), np.float64)
                    fg = np.zeros((1, 65), np.float64)
                    cv2.grabCut(bgr, gc_mask, (x1, y1, x2 - x1, y2 - y1),
                                bg, fg, 5, cv2.GC_INIT_WITH_RECT)
                    grabcut_iou, grabcut_dice = score(np.isin(gc_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]), gt)
                except cv2.error:
                    pass
            rows.append([image_id, a["id"], a["category_id"], sam_iou, sam_dice,
                         grabcut_iou, grabcut_dice])
            if len(rows) >= args.max_objects:
                break
        if len(rows) >= args.max_objects:
            break
    if not rows:
        raise RuntimeError("No scored examples: fail instead of producing fictitious metrics")
    header = ["image_id", "annotation_id", "category_id", "sam_iou", "sam_dice",
              "grabcut_iou", "grabcut_dice"]
    with open(args.out / "per_object.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    sam_mean = float(np.mean([r[3] for r in rows]))
    paired = [r for r in rows if r[5] != ""]
    result = {
        "status": "trained_model_inference_executed",
        "model": "Meta SAM ViT-B (original checkpoint)", "device": "cpu",
        "dataset": "COCO 2017 validation", "protocol": "GT bounding-box oracle prompts",
        "is_official_coco_ap": False, "seed": args.seed,
        "selected_image_ids": selected,
        "scored_images": len({r[0] for r in rows}), "scored_instances": len(rows),
        "sam_mean_iou": sam_mean,
        "sam_mean_dice": float(np.mean([r[4] for r in rows])),
        "paired_instances": len(paired),
        "sam_paired_mean_iou": float(np.mean([r[3] for r in paired])) if paired else None,
        "grabcut_paired_mean_iou": float(np.mean([r[5] for r in paired])) if paired else None,
        "grabcut_failure_count": len(rows) - len(paired),
        "checkpoint_sha256": sha256(checkpoint),
        "annotations_sha256": sha256(annotation_path),
        "image_sha256": images, "torch_version": torch.__version__
    }
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("selected_image_ids", "image_sha256")}, indent=2))


if __name__ == "__main__":
    main()
