# Chapter 5 — implementation of proportional book-wide scientific certification

Policy adopted: [SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md](../../../../../SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md). The canonical book-wide file is located at the repository root. This report applies the new policy to Chapter 5 without changing earlier authentic experimental findings.

## Decision

**Level assignments:** 9 Level B experimental applications/case studies and 2 Level C flagship/research units (CS04 cross-dataset autonomous driving and CS05 production autonomous driving stack). **Level A:** none of the 11 currently titled applications or case studies. A technical demonstration can be Level A only if it is independently and explicitly titled as a focused demonstration, not by relabeling an unexecuted application.

**Scientific certification:** **HOLD — 0/11 newly verified fully certified by this transition.** The existing strict per-requirement register in `CH5_R1_R10_EVIDENCE_MATRIX_20261112.csv` remains an unchanged evidence snapshot; its historic 6 PASS / 32 PARTIAL / 72 FAIL classification is not silently promoted to PASS because requirements were reinterpreted. A fresh evidence-by-evidence proportional reassessment must be done for each requirement with source artifacts and independently accepted methods before a new score is assigned.

## Level details

- **B:** CS01 KITTI 3D; CS02 agriculture UAV; CS03 YOLO drone; CS06 camera–LiDAR fusion; CS07 fastMRI; CS08 NeRF versus 3D Gaussian Splatting; CS09 SAM; CS10 highway lanes; CS11 pedestrian occlusion.
- **C:** CS04 cross-dataset autonomous-driving evaluation; CS05 production autonomous-driving stack.
- A Level B offline application may use genuine independent testing rather than physically deploying a vehicle or conducting a clinical trial **provided its title and conclusions only promise offline performance**. CS03's existing autonomous title and claims cannot bypass its actual tracking/control gap.
- Level C requires representative authentic domain/system evaluation as inherent to its title, not merely a diagram or one-source test.

## Specific proportional interpretations

**R2:** For frozen pretrained inference, document the original weights/training-provenance and the actual independent evaluation set with leakage considerations; do not fabricate train/validation/test partitions if none were trained locally. CS09 and CS03 can be reassessed for R2 on this basis only after documenting what original pretraining overlap and sample selection can be independently established.

**R5:** A matched quantitative baseline is mandatory for comparative titles like NeRF vs 3DGS, for cross-dataset claims and for claimed improvements. For a single-method explanatory unit, a quantitatively justified analytic/physical reference is acceptable when it actually measures the task. SAM's original GrabCut same-box comparison is valid within its 99 eligible masks, but automatic Mask R-CNN vs oracle-prompt SAM remains an unequal-input exploratory comparison.

**R6:** Use relevant statistical methods: original-image bootstrap for grouped scenes, subject-level split for medical images, appropriately repeated runs for stochastic models, or numerical residual/sensitivity methods when deterministic. Report samples and incomplete classes, and never manufacture intervals from aggregate-only data.

**R7:** An offline authentic benchmark can validate an offline model application. Physical drone flight, real-vehicle safety or clinical benefit is only required for claims that actually promise those outcomes. A pipeline claiming **production autonomous operation** still needs observed integrated stack evidence.

**R8:** Authentic result figures remain mandatory; avoid habitual bar charts. The measured VisDrone dot plot and measured SAM interval plot already have partial standalone figure-level audits, but the integrated chapter PDF assets are not fully inspected.

**R9:** Complete original executable code and configuration, exact checkpoints and their hashes (or verified authorized retrieval instructions), authentic data identifiers, actual per-scene outputs and figure generation must be evidenced in GitHub or a documented preserved workflow artifact. Controlled external data cannot be redistributed without authorization.

**R10:** Tighten claims rather than overclaiming. If a named chapter application promises occlusion, fusion, tracking, domain transfer or neural rendering comparison, missing that application-specific component is still an unsatisfied R10 and likely R3/R4 even under Level B.

## Concrete unresolved experimental inputs

- CS01: authorized original KITTI data and compatible PointPillars checkpoint, official AP and real inference latency.
- CS02: agricultural field observations, labels and task-matched detector inference.
- CS03: real video identities/tracks and measured control/planning outcomes.
- CS04: real domain-separated driving datasets, named-model runs and paired baseline.
- CS05: real integrated stack traces and task-relevant operational/safety checks.
- CS06: synchronized calibrated camera-LiDAR frames and original fusion results.
- CS07: access-controlled original fastMRI data and trained reconstruction model output.
- CS08: real posed scenes, matched Nerfacto/3DGS runs and measured resource/performance.
- CS09: more representative original real masks, prompt-robustness and appropriately fair comparisons; existing original COCO evidence remains genuine.
- CS10: original TuSimple images/labels and matched pretrained model inference, official lane scoring when claimed.
- CS11: original occlusion-annotated CrowdHuman/CityPersons model predictions and stratified errors.

## Implementation and QA files

- Root: `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md`.
- Chapter-level assignments: `CH5_PROPORTIONAL_LEVEL_ASSIGNMENTS_20261112.csv`.
- Automated coverage/scoping checker: `validate_proportional_levels.py`.
- Historical experiment statuses retained: `CH5_R1_R10_EVIDENCE_MATRIX_20261112.csv`.
- Strict status checker retained: `validate_r1_r10_matrix.py`.

This is a **standard and register implementation**, not a claim of new model runs or manuscript figures. A future full chapter release requires re-execution/verification and a complete current manuscript PDF render/figure inspection. No unit's current missing empirical evidence has been waived.
