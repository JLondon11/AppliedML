# Chapter 5 — Segment Anything Model (SAM) on original COCO real data

**Experiment status: executed and independently verified on a documented small COCO subset; full-chapter Master QA remains BLOCKED.** The original Meta SAM ViT-B checkpoint was executed on three original COCO 2017 validation photographs with eight original instance masks. Successful GitHub Actions run: https://github.com/JLondon11/AppliedML/actions/runs/41979092197.

**Actual measured results:** mean SAM per-instance IoU=0.8628225175; mean Dice=0.9251755029 (8 objects); paired SAM IoU=0.8747302910 vs GrabCut IoU=0.5260045809 (6 objects). The remaining two boxes were invalid for the rectangle-based GrabCut baseline and are excluded from the paired comparison.

## Exact reproducibility protocol
- Model: official Meta SAM ViT-B checkpoint `sam_vit_b_01ec64.pth`, SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`. Official code https://github.com/facebookresearch/segment-anything. No fine-tuning.
- Dataset: original COCO 2017 validation images and instance masks, image IDs **139,632,776**. Original annotations SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` (full COCO annotations archive in `selection_provenance.json`). Original archive SHA256: annotations `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`; val images `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`. Both downloaded over HTTPS from a pinned version of an original-archive mirror and hash-checked.
- Image selection: ascending image ID among images with >=2 annotated non-crowd objects of area>=3000; limit to 3 images, 8 eligible objects in original annotation iteration order. These **nonrandom** images are not an official benchmark sample.
- Prompts: oracle ground-truth COCO bounding boxes, identical prompts to original Meta SAM and OpenCV GrabCut, `multimask_output=False`.
- Metrics: local reference binary-mask IoU and Dice. Six paired SAM/GrabCut comparisons; separate all-8 SAM mean. The result **must not be called COCO AP**, SAM automatic no-prompt performance, or a statistically representative full benchmark.
- Environment: PyTorch 2.15.0+cpu. The original model and masks were actually used to generate source-locked CSV, JSON, and visual PNG/PDF/SVG panels.
- Raw verified result files: `results/coco_vitb_3images/`. Full generated figure and metrics artifacts from GitHub Actions workflow `chapter5-original-SAM-ViTB-real-COCO-inference`.
- Manuscript integration: revised 102-page Chapter 5 contains Table 5.13 (real mask overlap) and Figures 5.46–5.47 (genuine prediction panels and paired IoU scatter). The prior actual GrabCut photograph-only baseline remains Figure 5.48, clearly distinguished from the model.
- **Remaining approval work:** replicate a sufficiently large, preregistered COCO evaluation sample or official protocol; matched supervised segmenter, prompt sensitivity, confidence intervals, latency, and any target-domain autonomous perception test.

## Scripts
`prepare_real_coco_subset.py` reads original annotation archive and extracts original JPEG bytes from the verified original val2017 archive. `evaluate_sam_coco.py` loads SAM and original masks, computes per-object metrics, and writes comparative figures. `test_local.py` tests computational logic but cannot substitute for the recorded, successfully executed model run.

```bash
# In GitHub Actions, the workflow downloads and checksum-verifies the original
# data and weight archives, selects exactly three images, and executes:
python evaluate_sam_coco.py --coco-json /tmp/ch5_sam/subset/instances_subset.json \
  --images /tmp/ch5_sam/subset \
  --checkpoint /tmp/ch5_sam/weights/sam_vit_b_01ec64.pth \
  --model-type vit_b --device cpu --max-objects 8 --max-plots 4 \
  --min-box-area 3000 --output results/
```
