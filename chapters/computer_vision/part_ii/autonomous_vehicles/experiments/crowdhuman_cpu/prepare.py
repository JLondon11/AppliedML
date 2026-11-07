"""Prepare licensed original CrowdHuman validation subset for CPU model inference.
Image data are ephemeral and must NOT be published as GitHub workflow artifacts.
"""
import argparse
import json
import random
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
)

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=Path("crowdhuman_data"))
p.add_argument("--images", type=int, default=24)
p.add_argument("--seed", type=int, default=20261106)
a = p.parse_args()
if not 1 <= a.images <= 4370:
    p.error("Images must be between 1 and 4370")
a.out.mkdir(parents=True, exist_ok=True)
repo = "Funtion/CrowdHuman"
annotation_file = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename="annotation_val.odgt"))
records = [json.loads(s) for s in annotation_file.read_text().splitlines() if s.strip()]
if len(records) != 4370:
    raise RuntimeError(f"Expected 4370 original validation annotations, got {len(records)}")
records.sort(key=lambda x: x["ID"])
sample = random.Random(a.seed).sample(records, a.images)
archive = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename="CrowdHuman_val.zip"))
images = a.out / "images"
images.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    mapping = {}
    for member in z.infolist():
        if member.is_dir() or not member.filename.lower().endswith(".jpg"):
            continue
        basename = Path(member.filename).name
        if basename in mapping:
            raise RuntimeError(f"Ambiguous duplicate image name: {basename}")
        mapping[basename] = member
    for record in sample:
        name = record["ID"] + ".jpg"
        if name not in mapping:
            raise FileNotFoundError(f"Missing original CrowdHuman validation image {name}")
        data = z.read(mapping[name])
        (images / name).write_bytes(data)
with (a.out / "subset.odgt").open("w") as f:
    for record in sample:
        f.write(json.dumps(record) + "\n")
torch.set_num_threads(4)
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).eval().cpu()
weights_path = a.out / "fasterrcnn_resnet50_fpn_coco_state_dict.pth"
torch.save(model.state_dict(), weights_path)
print(json.dumps({"dataset":"Original CrowdHuman val","source":repo,
                  "total_validation_images":len(records),
                  "selected_images":len(sample),"seed":a.seed,
                  "checkpoint":"Torchvision COCO Faster R-CNN ResNet50 FPN"},indent=2))
