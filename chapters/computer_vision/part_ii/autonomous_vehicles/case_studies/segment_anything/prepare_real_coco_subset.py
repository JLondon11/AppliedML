#!/usr/bin/env python3
"""Prepare original COCO val2017 image+instance subset for actual SAM inference.

Reads the OFFICIAL COCO annotations JSON, retains only original annotations, and
downloads original image bytes. Does not invent masks/annotations.
Selection: ascending image ID, >=2 eligible original instances per image, area >=3000,
noncrowd; first 3 such images. All eligible annotations per image retained.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def choose(data, count=3, min_area=3000):
    candidates = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) or ann.get("area", 0) < min_area:
            continue
        if "segmentation" not in ann:
            continue
        candidates.setdefault(ann["image_id"], []).append(ann)
    images = []
    for im in sorted(data["images"], key=lambda x: x["id"]):
        eligible = candidates.get(im["id"], [])
        if len(eligible) >= 2:
            images.append(im)
            if len(images) >= count:
                break
    if len(images) != count:
        raise ValueError("Official source does not contain required sample")
    selected = set(im["id"] for im in images)
    anns = [a for a in data["annotations"] if a["image_id"] in selected]
    # Keep ALL annotations for selected images to retain the original evidence.
    result = {k: data[k] for k in ("info", "licenses", "categories") if k in data}
    result.update(images=images, annotations=anns)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=3)
    a = p.parse_args()
    if not a.original_json.is_file():
        raise FileNotFoundError("Original COCO instances_val2017.json missing")
    d = json.loads(a.original_json.read_text())
    selected = choose(d, a.count)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for im in selected["images"]:
        file = a.output_dir / im["file_name"]
        url = "https://images.cocodataset.org/val2017/" + im["file_name"]
        urllib.request.urlretrieve(url, file)
        if file.stat().st_size < 10000:
            raise ValueError(f"Original source JPEG suspiciously small: {file}")
    (a.output_dir / "instances_subset.json").write_text(json.dumps(selected))
    report = {
        "original_annotations_sha256": hashlib.sha256(a.original_json.read_bytes()).hexdigest(),
        "image_ids": [int(im["id"]) for im in selected["images"]],
        "original_image_sha256": {
            im["file_name"]: hashlib.sha256((a.output_dir / im["file_name"]).read_bytes()).hexdigest()
            for im in selected["images"]
        },
        "selection": "ascending COCO image ID, >=2 noncrowd segmented objects with area>=3000",
        "scope": "small nonrandom sample from genuine val2017; NOT full COCO test set"
    }
    (a.output_dir / "selection_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
