# Chapter 5 — independent Level A real-data detection demonstrations

Date: 2026-11-12. Governing policy: root `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md` (scientific vs. editorial/production certification separated).

## Scientific decision

**Two executed real-data demonstration units certified at Level A (10/10 applicable R1–R10 each), without changing the 11-parent-application matrix:**
- **CV5-D01 — Offline aerial-image object detection using COCO-pretrained YOLO11n on original VisDrone:** `SCIENTIFIC_CERTIFIED`, `PRODUCTION_HOLD`.
- **CV5-D02 — Offline pedestrian detection using COCO-pretrained Faster R-CNN on original Penn–Fudan photographs:** `SCIENTIFIC_CERTIFIED`, `PRODUCTION_HOLD`.

**Parent CV5-CS03 (YOLO autonomous drone) and CV5-CS11 (pedestrian detection under occlusion) remain SCIENTIFIC_HOLD.** Their titles require tracking/control and occlusion-stratified evidence, respectively. Together with separately Level B-certified original-box-prompt SAM CV5-CS09, this is **three scientifically certified executed experiments (one full named CS unit and two embedded Level A demos), but still only 1/11 full named parent applications certified**.

## Real experimental source audit

The original successful GitHub Actions records and **actual unexpired ZIP experiment artifacts** were retrieved and read in the current runtime.

| Original experiment | Actions run | Original artifact ID | ZIP SHA256 | Recorded original image files | Unique source image SHA256 | Genuine archived output figure triples |
|---|---:|---:|---|---:|---:|---:|
| YOLO11n on original VisDrone2019-DET-val | 41942041254 | 13377330273 | `c1b2cc01b14f20f93ed6d37d5bca60e7146980bb34b9558a25f993366bb6b36c` | 24 | **23** | 5 PNG/PDF/SVG triples |
| torchvision Faster R-CNN MobileNetV3 320 FPN on original PennFudanPed | 41980566831 | 13393014885 | `1bc0a2e7690c47c4a51a7fd1b09603134d7690cfb78bccab9dae22cb7f353fb5` | 170 | **170** | 6 PNG/PDF/SVG triples |

Each source ZIP includes the original model-run provenance, input-image SHA256 inventory, original model checkpoint SHA256, original predictions/detections CSV, summary metrics CSV, and real-result photographed inference-overlay figures. The eleven PNG panels were inspected together in a local contact sheet. The figure originals are meaningful real-image comparisons; final printed page aesthetics must be separately reviewed.

**Original data anomaly discovered, disclosed, and required in all future reporting:** VisDrone images `0000022_00000_d_0000004.jpg` and `0000023_00000_d_0000008.jpg` have **identical original SHA256 `0bea9d13456e6aa7a92ad1af21a89960eb66916855ed8a2c3d0371b1d7d37bd0`**, despite different file names and different annotation-file SHA256. Therefore describe the experiment as **24 evaluated original files, 23 distinct image contents**. No attempt is made to fabricate a new 23-file benchmark metric by deleting a record without rerunning its GT matching. Do not claim 24 independent scenes.

## CV5-D01 (Level A): verified genuine single-frame YOLO outputs

- Source experiment: official VisDrone original photographs and annotations, pretraining COCO and original Ultralytics YOLO11n checkpoint SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`, no fine-tuning.
- Original source archive directly contains **2,589 retained detection rows**, 24 source-file identifiers and 24 source image SHA256s, all box coordinates/scores valid in an independent recorded-output check.
- Per-class source CSV records **972 ground-truth object annotations**, **322 true-positive matches** under the original local protocol. Original local macro AP50 **0.1406456330251602** was independently re-averaged from the **five nonempty classes**. The absent bus GT category was correctly excluded (undefined AP, not numerical zero).
- The results are a selected-file, COCO-to-VisDrone category-mapped transfer **illustration**, not official VisDrone AP, multi-frame identity tracking, flight control, or an agricultural obstacle detector.
- Level A R5 is satisfied through real per-object VisDrone annotated reference truth and measured class AP50. It does **not** pretend to be a matched-head-to-head architecture superiority claim. R6 uses descriptive matched TP/error counts appropriate to the limited demonstration; no representative population confidence intervals or independent scenes claim.
- **Level A scientific status: 10 PASS / CERTIFIED only at this explicitly narrow scope.** Production remains HOLD.

## CV5-D02 (Level A): verified genuine pretrained pedestrian output

- Original PennFudanPed 170 photographed source-file hashes are **all distinct**; original checkpoint SHA256 `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`. No fine-tuning or invented training splits.
- Original source ZIP contains **1,199 actual retained person bounding-box predictions** with valid coordinates/scores and named original photograph IDs; original model evaluator reports **423 original annotated persons, 417 confidence-ranked IoU>=0.5 true-positive matches**, and local AP50 **0.9681834843380837**.
- Independently recalculated `417/423 = 0.9858156028` recall and `417/1199 = 0.3477898249` final cumulative precision to recorded numerical precision. **The original ground-truth mask contents are not included in the archive, therefore the local AP50 is faithfully identified as the original executed evaluator's result, not as an independently recomputed official benchmark score.**
- The six genuine original photographic overlay figure triples are available and visually surveyed. No occlusion-stratified labels or performance have been verified.
- Level A R5 uses original Penn–Fudan annotations as the quantitative reference for basic detection; there is **no claim of comparative architecture superiority or occlusion robustness**. R6 is appropriate descriptive original detection error counts.
- **Level A scientific status: 10 PASS / CERTIFIED only at this explicitly narrow scope.** Production remains HOLD.

## What R2/R5/R6/R7/R9 mean at Level A

- **R2:** frozen checkpoint without retraining: source selections and any known duplicated observations are documented; pretraining/evaluation overlap is not falsely assumed absent. No fictional train/val split is required.
- **R5:** original task-reference annotations and actual IoU/TP/AP evaluation satisfy a **single-algorithm real-data demonstration**; comparative title/claims would need a matched comparator.
- **R6:** reported genuine model-error counts, precision/recall and measured AP meet focused Level A inference demonstration depth; cluster bootstrap or multi-seed retraining are not prerequisites.
- **R7:** real photographs in the intended offline evaluation domain suffice for **offline image detection**, not flight/real-time/vehicle safety or occlusion engineering claims.
- **R9:** repository source scripts, command/run references, identified exact original checkpoint weights, recorded original source image hashes, original machine-readable prediction/metric CSVs and original figure outputs satisfy proportional scientific traceability; no universal requirement to store copyrighted raw source datasets or finished Chapter 5 PDF on GitHub. Original GitHub Actions archives have finite retention, so long-term numerical-record preservation remains a production/release resilience improvement, not a precondition to this narrow scientific certificate.

## Scope and production separation

The original chapter labels and all 11 parent case studies are **unchanged**. Their scientific statuses remain: CS09 original SAM **CERTIFIED B**, ten other parent units **HOLD**. These two **newly and explicitly named embedded Level A demonstrations** do not silently revise any parent R1–R10 row. The actual chapter PDF/TeX was not accessible for final page-placement/caption/print-size approval; **all manuscript production statuses remain HOLD**.

Source-of-record: `qa/CH5_LEVEL_A_EMBEDDED_DEMONSTRATION_CERTIFICATIONS_20261112.csv`. Reproducible independent ZIP/measurement contract checker: `qa/verify_level_a_original_demonstrations.py`. The verifier does not falsely assert independent AP recomputation without original ground-truth files.
