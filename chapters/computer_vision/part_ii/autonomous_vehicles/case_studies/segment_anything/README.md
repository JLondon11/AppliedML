# Chapter 5 · Segment Anything Model (SAM) — real COCO masks

**Master QA status: BLOCKED (model inference not run in current environment).**

This directory now contains an executable **original Meta SAM checkpoint** evaluation, generating per-instance and per-category results tables, an IoU comparison plot, and qualitative scientific panels from original COCO photographs and ground-truth instance masks. These are generated **only after genuine SAM inference**. The missing model checkpoint and COCO images are never replaced with generated or fabricated outputs.

## Primary sources and exact protocol

- **Model:** Kirillov et al., *Segment Anything* (ICCV 2023); [official implementation, checkpoint links and instructions](https://github.com/facebookresearch/segment-anything). Use the correct ViT-B/ViT-L/ViT-H checkpoint for `--model-type`.
- **Dataset:** Lin et al., *Microsoft COCO: Common Objects in Context* (ECCV 2014); [official COCO data](https://cocodataset.org/#download). Download actual `val2017` images and `annotations_trainval2017.zip` for `instances_val2017.json`.
- **Prompts:** each eligible *ground-truth* COCO instance bounding box is used as a SAM oracle box prompt; **this is not automatic segmentation or prompt-free prediction**.
- **Metrics:** per-instance binary mask IoU and Dice. GrabCut uses the same prompt and image, with invalid border-touching prompts excluded from the paired comparison. The output is **not** COCO AP or an official SAM benchmark.
- **Reproducibility:** original checkpoint SHA-256, ground-truth annotation JSON SHA-256, model type, PyTorch version, sample count and protocol recorded in JSON. Dataset is not checked into GitHub.

## Running

```bash
pip install torch torchvision opencv-python matplotlib numpy pycocotools
pip install git+https://github.com/facebookresearch/segment-anything.git
python test_local.py
python evaluate_sam_coco.py \
  --coco-json /data/coco/annotations/instances_val2017.json \
  --images /data/coco/val2017 \
  --checkpoint /models/sam_vit_b_01ec64.pth \
  --model-type vit_b --device cuda \
  --max-objects 80 --output results/sam_vit_b
```

Output paths: `per_object_metrics.csv`, `per_class_summary.csv`, `metrics_summary.json`, and `Figures/sam_coco_*.{png,pdf,svg}`, `Figures/sam_vs_grabcut_iou.{png,pdf,svg}`.

**Scientific caveat:** a small, filtered sample of large non-crowd COCO objects and an oracle prompt do not estimate zero-shot segmentation under naturally chosen prompts. Document the sample/split and selection policy before comparing any model to published results.

## State of verification, November 3 2026

The existing chapter includes a real-photograph GrabCut demonstration, **not SAM inference**. Local algorithm tests of bounding-box conversion, mask IoU/Dice, baseline segmentation and hash recording passed. The official SAM package and model checkpoint are not locally installed; official COCO validation images and annotation JSON are absent. Thus no SAM checkpoint-backed numerical results or output figures were obtained and the chapter's SAM study is **not Master QA approved**.
