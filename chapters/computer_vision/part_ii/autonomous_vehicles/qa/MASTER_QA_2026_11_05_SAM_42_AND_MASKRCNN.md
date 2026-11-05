# Chapter 5 — Real SAM/Mask R-CNN experiment expansion, 2026-11-05

**Chapter identity:** Chapter 5, Computer Vision Part II — Autonomous Systems.

## Genuinely executed original models and original annotation evidence

All workflows below were executed, concluded **success**, and retained their original numerical CSV, provenance and PNG/PDF/SVG artifacts:

1. [Official Meta SAM ViT-B on 18 seeded original COCO validation photos](https://github.com/JLondon11/AppliedML/actions/runs/41980537230): 59 original instances, mean oracle-box mask IoU 0.7773286118, mean Dice 0.8578804397. GrabCut has 35 valid matched rectangle prompts.
2. [Official Meta SAM ViT-B on 24 earliest eligible original COCO validation photos](https://github.com/JLondon11/AppliedML/actions/runs/41981250063): 88 original instances, mean oracle-box mask IoU 0.7794836658, mean Dice 0.8575854773. GrabCut has 64 valid matched rectangle prompts.
3. [Official COCO-pretrained torchvision Mask R-CNN ResNet50 FPN](https://github.com/JLondon11/AppliedML/actions/runs/41981859101): on **exact same 18 photos and 59 original annotation IDs** as seeded SAM, 56 predicted class/box matches, mean instance mask IoU 0.7453008746. Unmatched GT instances count as zero. No oracle boxes were supplied to Mask R-CNN.

The first two expanded SAM samples have **zero overlapping image IDs**. Their original image subsets together contain 42 photos and 147 original annotated objects. The earliest 3-image pilot was already included inside the ascending 24-image group and was not counted twice.

## Independently computed pooled and model-comparative results

Cluster-resampling analysis is deterministic (10,000 resamples of original scene IDs, seed 20261105), preserving grouping of objects by original photograph. Verified pooled SAM/GrabCut statistics:

| Metric | N objects | Observed mean | Exploratory scene-cluster 95% interval |
|---|---:|---:|---|
| SAM original mask IoU | 147 | 0.7786187121 | [0.7463495294, 0.8090720731] |
| SAM original mask Dice | 147 | 0.8577038635 | [0.8304908712, 0.8822431518] |
| SAM mask IoU, paired valid objects | 99 | 0.8062725696 | [0.7734335799, 0.8398754525] |
| GrabCut mask IoU, paired valid objects | 99 | 0.4864770977 | [0.3841488706, 0.6034062431] |
| SAM minus GrabCut paired mask IoU | 99 | 0.3197954719 | [0.2135099225, 0.4130896346] |

Mask R-CNN automatic inference vs oracle-prompt SAM on **exact same 59 original instances**: Mask R-CNN mean mask IoU 0.7453008746, SAM mean mask IoU 0.7773286118, mean paired difference 0.0320277372, scene-cluster exploratory 95% interval **[-0.0241727693, 0.0916450326]**. Because zero lies in the interval, these results do not demonstrate a robust advantage; more importantly, **the methods have unequal input information**, so the observation cannot be considered an equal-conditions architecture ranking.

Original checkpoint SHA-256: SAM `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`; pretrained Mask R-CNN `bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0`.

## Manuscript integration

The delivered **108-page** compiled Chapter 5 `Chapter_5.pdf` and its `Chapter_5.tex` contain:
- An expanded Table **5.15** with original SAM 42-image / 147-object pooled metrics and conditional confidence intervals.
- A real-data 42-scene reproducibility/uncertainty figure.
- Original SAM and trained Mask R-CNN same-scene comparison Table **5.16** and two further real-result figures (paired scatter/cluster bootstrap and original COCO cat photograph with trained Mask R-CNN mask).
- Original annotations, metrics, checkpoint references, multiple explanatory paragraphs and clear warnings against official COCO AP/zero-shot semantic conclusions.
- Final LaTeX build: two successful `pdflatex` passes, 108 pages, no unresolved citations/cross-references or overfull lines. Scientific figures inspected on rendered PDF pages 87–88.

The original local TeX/PDF and figure assets are distributed in a user downloadable chapter release ZIP; **no claim that binary manuscript artifacts were committed to GitHub**.

## Publication gates still blocked

Neither sample is statistically representative of all COCO annotation sizes/classes because only non-crowd objects with original area >=3000 pixels from images satisfying the two-object criterion were eligible. The scene-cluster intervals are conditional exploratory summaries. SAM uses **oracle ground-truth bounding-box prompts** whereas Mask R-CNN performs automatic category detection and segmentation; comparison is not equal-information. Neither is official COCO segmentation AP. Prompt robustness, automated detector-to-SAM integration, independent test splits, full labeled benchmark evaluation, target-domain robustness, runtime and the other Chapter 5 named applications still need executed experiments. **Chapter 5 Master QA NOT APPROVED.**

Reproducible scripts: `case_studies/segment_anything/analyze_sam_42_images.py`, `compare_maskrcnn_sam.py`, `evaluate_maskrcnn_coco_same_scene.py`, plus executed original SAM runner and GitHub Actions workflows.
