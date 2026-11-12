# Chapter 5 — experimental execution readiness and numerical correctness, 2026-11-11

**Scope:** TuSimple lane detection and CrowdHuman occluded pedestrian detection, the next two named experiments. **Model-execution status: BLOCKED**, not completed; **Chapter 5 certification: HOLD**.

## Inputs checked

- Original TuSimple images / label_data JSONL and a compatible lane-mask ONNX checkpoint are not present in the current execution workspace.
- Original CrowdHuman images / validation ODGT and the exact compatible torchvision Faster R-CNN state dictionary are not present in the current execution workspace.
- The repository contains a local evaluator for each task, but runnable code and unit tests are not evidence of real-data trained-model inference. No new checkpoint inference was executed in this pass.
- Existing separate Penn–Fudan, VisDrone, and COCO results retain their previously verified partial-evidence status; they are not substituted for TuSimple or CrowdHuman.

## Code correctness fix completed

The TuSimple ONNX script previously inferred probability-versus-logit output semantics from the observed numerical range; **a legitimate all-zero-logits output is in [0,1] and would be wrongly read as probability zero rather than sigmoid(0)=0.5**. This affects mask classification at the default 0.5 threshold. The implementation now requires `--output-semantics logits|probability`, explicitly applies sigmoid only for logits, rejects out-of-range probabilities, and records semantics in its metrics manifest. The README usage and unit tests were amended.

Commits on `main`:
- `198cde84a40a3a636e0517cc78c2a7122af51b2e` implementation
- `0fd61010e67465ad997cfaeb9048d161dc2ac568` regression tests
- `e82bf0c4b69ec69a3c0e83554f0c98d6c798840c` README

A locally reconstructed copy of the changed mask-conversion function passed the four targeted semantics checks using actual NumPy/OpenCV; that is **not a claim that GitHub CI executed the repository test suite**.

## Scientific gates for actual execution

**TuSimple:** obtain authorized original images and annotation JSONL, validated lane-specific checkpoint with documented preprocessing and output semantics, fixed train/validation/test partitions and exact image counts. Run all benchmark annotations with saved predictions and original-image overlays; calculate the existing 4-pixel-polyline local scores and, for official claims, separately implement and execute the published TuSimple evaluator. Include failures, legibility and domain-specific metrics rather than claiming official accuracy from pixel IoU.

**CrowdHuman:** obtain authorized original validation JPEG/ODGT and exact matching COCO Faster R-CNN weights; execute model on all selected original annotations, record image counts/GT instances/weights hashes and prediction lists, present full-body recall by visible-to-full box area strata and actual overlays. Use the official CrowdHuman evaluator for any official AP/MR/JI claim, and retain a visible limitation that bounding-box area is not a pixel-accurate occlusion label.

## Release status

No new empirical model outputs, named-dataset plots, or revised Chapter 5 PDF were produced. Source and tests were committed. Do not mark CS10 or CS11 certified until actual named dataset/checkpoint model runs, scientific figures, manuscript linkage, and independent PDF visual review are present.
