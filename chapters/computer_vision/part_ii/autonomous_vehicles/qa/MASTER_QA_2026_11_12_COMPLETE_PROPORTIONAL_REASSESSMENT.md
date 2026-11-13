# Chapter 5 — complete proportional R1–R10 scientific reassessment

**Date:** 2026-11-12. Governing policy: repository-root `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md`. Source of this fresh review: `CH5_PROPORTIONAL_R1_R10_FULL_REASSESSMENT_20261112.csv`.

## Outcome

**Zero of eleven units certified.** The newer proportional evidence statuses are **11 PASS, 27 PARTIAL and 72 FAIL across 110 requirements**, compared with the preserved older strict snapshot (6 PASS, 32 PARTIAL and 72 FAIL). The difference is **five proportional reinterpretations to PASS**, not five newly executed experiments; do not present it as scientific performance improvement.

The current Chapter 5 manuscript TeX/PDF is absent in the active workspace and the complete connected GitHub tree. A prior chapter release is mentioned in old QA reports, but its latest actual PDF/layout cannot be inspected by relying on reported descriptions.

**Levels:** 9 B experimental units, 2 C research or production units (CS04 and CS05). There are no A-level entries among the unchanged full application/case-study titles. No unsupported N/A was applied.

## Complete unit-specific findings and acceptance evidence

| Unit | Level | Observed evidence and current scope | Scientific evidence required for certification |
|---|---|---|---|
| CS01 KITTI PointPillars | B | Validated runner, LiDAR/calibration/label preflight and result renderer; no named trained inference | Original authorized KITTI LiDAR/labels/split and compatible pretrained weights; execute PointPillars on original validation, retain per-frame boxes/GT, actual class AP and measured inference latency; publication-quality BEV predictions and error plots; matched baseline/reference |
| CS02 UAV precision agriculture | B | Existing VisDrone aerial people/vehicle transfer genuine, but agricultural field detector not executed | Original annotated agricultural-field obstacles and compatible trained model, field-specific categories and measured outputs/baselines, domain failures and real-output figures; no automatic extrapolation from urban aerial photos |
| CS03 YOLO autonomous drone | B | Original 24 VisDrone frames with YOLO11n single-image AP; measured figure | Named title requires real chronological video identities, validated track/identity metrics and action/flight-planning traces; matched reference and realistic latency/operational scope. Alternatively retitle/restrict to *single-frame aerial object-detection transfer* and conduct fresh R1–R10 evaluation; no quiet certification of unchanged title |
| CS04 cross-dataset autonomous driving | C | Generated-data protocol exercise only | Two genuine non-overlapping, harmonized real driving datasets, actual model predictions on both, matched baselines, sample counts, domain-shift statistics, uncertainty/failure examples, directly generated scientific figures |
| CS05 production autonomous-driving stack | C | A telemetry CSV evaluator with software fixtures, no recorded stack | Authentic synchronized multi-stage stack traces and actual model/actuator outputs, independent ground truth, latency distributions, intervention/error/robustness analysis with declared hardware; do not substitute fabricated telemetry |
| CS06 camera–LiDAR multimodal | B | Genuine Middlebury stereo-photo baseline but not fusion | Original synchronized calibrated image/LiDAR scenes; executed fused detector, retained point-cloud BEV/photo overlays, ablation against single sensor, matched metrics and alignment validity |
| CS07 deep-learning accelerated MRI | B | Prepared fastMRI HDF5 classical comparator and numerical fixtures | Authorized original fastMRI k-space, executed compatible trained VarNet/Adaptive-CS-Net (or accurately titled alternative), subject-disjoint samples, matched reconstruction outputs, SSIM/PSNR and spatial residuals/limitations; clinical benefit not required for offline-only title |
| CS08 NeRF vs 3DGS | B | Two Nerfstudio command runners and separate real-photograph stereo baseline; no actual paired training | Genuine posed multi-view scene and disjoint evaluation views, both named trained reconstruction methods, matched PSNR/SSIM/LPIPS, measured training and rendering resource/throughput, real side-by-side views and failure/quality–speed Pareto |
| CS09 prompted SAM | B | Independently reconciled 42 original COCO scenes/147 masks with original per-object evidence; valid 99 same-box GrabCut paired comparisons and real-result figures. The correct scientifically limited scope is *frozen pretrained, ground-truth-box-prompted instance segmentation*. | Full latest final manuscript PDF/TeX with exact figure numbers/pixel sources, original figure-size/caption review and complete current code/configuration/asset lineage; durable source numerical evidence is now committed under `results/original_actions_preserved`, but image/figure source archival and actual final print review remain partially unverified |
| CS10 highway lane detection | B | Corrected original-sample lane-mask raster code, no trained TuSimple model inference | Original benchmark camera images/label split and compatible checkpoint, original predicted lane coordinates, appropriate official/local metrics, failure overlays and measured uncertainty; local raster IoU is not official TuSimple scoring |
| CS11 pedestrians under occlusion | B | Actual pretrained Faster R-CNN on 170 Penn–Fudan photos/423 persons with genuine detections; no occlusion-stratified validation | Original CrowdHuman/CityPersons full/visible-box annotations, executed model on source images, stratified recall/AP and matched baseline, crowded failure overlays, scene-cluster uncertainty; general Penn–Fudan detection cannot certify occlusion robustness |

## Requirement-specific proportional guidance

- **R2:** Frozen pretrained inference needs a documented independent evaluation sample, selection criteria, training-source overlap/unknowns and leakage controls, not invented model-training partitions. A source's literal test label is not proof the pretrained model never saw those images.
- **R5:** Comparative titles, model superiority claims and Level C transfer/production research need a properly matched quantitative reference. For a single-method offline demonstration, an actually computed physical/analytical reference can be enough. Never accept a paper's summary replot as a locally executed baseline.
- **R6:** Use task-appropriate uncertainty and failure metrics: clustered photographic scene intervals, subject-group medical study metrics, actual controller trace variability, or convergence/residual tests for deterministic physics methods. Do not invent seeds or confidence bounds.
- **R7:** Offline real-data evaluation is valid for an honestly offline task. Physical flight and clinical/vehicle safety trials are not universal necessities; the experimental unit cannot imply those untested deployment claims.
- **R8:** Real, original-model output figure mandatory for every named empirical unit, with appropriate scientific density, sample sizes, dimensions and legends; minimize bar charts and prefer overlays, residual maps, error distributions, matching point clouds or observed temporal traces. Only two plot groups were individually accepted in prior partial figure audits, not all final chapter graphics.
- **R9:** Original execution code, exact configuration, checkpoint/source identifier, per-image/per-object outputs, figures and final manuscript link are required. Restricted datasets and weights may have authorized access instructions and verified SHA256 rather than being redistributed; the code/outputs still need to be complete.
- **R10:** Cannot reclassify an application with tracking/control/occlusion/fusion/cross-dataset/real-time claims as complete based solely on a narrower but genuine experiment, unless the title and chapter conclusions are explicitly narrowed *and independently reassessed*.

## Data access barriers are not exemptions

Missing official KITTI point clouds/checkpoint, controlled-access fastMRI images, original TuSimple lanes, original CrowdHuman occlusion annotations, or suitable GPU/pose assets are legitimate evidence-production blockers **in the present runtime**, not reasons to mark R1/R3/R4/R8 or R9 N/A. No model predictions or experimental numerical results are simulated to fill gaps.

## New reproducible QA controls

- `CH5_PROPORTIONAL_R1_R10_FULL_REASSESSMENT_20261112.csv`: one row per unit, ten evidence statuses and unit-specific explanation.
- `validate_full_proportional_matrix.py`: checks all eleven unique unit IDs, B/C levels, mandatory empirical requirements, status values and explanation completeness. It will not claim a complete book science certificate from row labels alone.
- Historical matrix and three-unit fresh register retained unchanged as prior evidence snapshots.

**Final decision:** Complete proportionally interpreted Chapter 5 R1–R10 coverage has been audited and recorded. Scientific FULL CERTIFICATION = **0/11**, pending actual missing original-model experiments and final manuscript QA. The improved registry is not evidence of new model runs.
