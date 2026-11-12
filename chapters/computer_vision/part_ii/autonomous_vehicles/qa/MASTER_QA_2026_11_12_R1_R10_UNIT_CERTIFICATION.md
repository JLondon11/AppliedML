# Chapter 5 — strict R1–R10 unit scientific certification audit

Date: 2026-11-12. Chapter: Computer Vision Part II (autonomous systems).

**Decision: HOLD — 0 of 11 named units have demonstrated all ten scientific requirements.** This is an evidence-based decision, not a declaration that every existing experiment failed. The [unit-by-requirement status matrix](CH5_R1_R10_EVIDENCE_MATRIX_20261112.csv) is the authoritative audit register for this pass, with machine-checkable fail-closed policy in [validate_r1_r10_matrix.py](validate_r1_r10_matrix.py).

## R1–R10 definitions and passing conditions

- R1 authentic original dataset, source/license and verifiable provenance for the titled task.
- R2 correctly documented training/validation/testing or explicitly pretrained-only evaluation, held-out sample counts, and source-domain/sample leakage analysis.
- R3 actual execution of the specific named model or algorithm appropriate to the titled application.
- R4 retained real predictions, reconstruction outputs, complete trajectories or instrumented action traces (not only fixtures and diagrams).
- R5 at least one fair quantitative baseline under matched data, information budget and operating points, with honest differences stated.
- R6 meaningful task-specific metrics, uncertainty/cross-scene variability and failure/error analysis; appropriate protocol for official benchmark claims.
- R7 validation relevant to the real engineering/domain claim (deployment, safety, acquisition or robustness as applicable).
- R8 readable, aesthetically excellent, genuinely **measured-output** scientific figure; use overlays, point-cloud projections, paired error maps, uncertainty dot plots, temporal traces or performance–resource Pareto instead of routine bar charts.
- R9 all needed implementation/configuration/checkpoint identities, input and model hashes, per-instance evidence, regeneration steps, and source-to-manuscript figure traceability.
- R10 conclusions no broader than evidence and **the entire titled application** (do not certify autonomous control from single-image detection).

**PASS** requires demonstrated requirement-specific evidence. **PARTIAL** means genuine relevant evidence exists but the requirement is incomplete. **FAIL** means no sufficient support for that requirement in the reviewed evidence; it is not a claim that the task is impossible. **NOT VERIFIED** is reserved for evidence that may exist but cannot be inspected. None is equivalent to PASS for certification.

## Verified evidence and blockers by unit

| Unit | Real evidence actually supported | Specific absent evidence preventing full certification |
|---|---|---|
| CS01 KITTI 3D PointPillars | Original-model execution wrapper, strict full-split LiDAR/calibration/label preflight and BEV renderer are committed; fixtures only | Obtain legal KITTI point clouds/calibrations/labels and compatible pretrained PointPillars model; execute official OpenPCDet AP by difficulty and measured batch/device latency; retain original predictions and compare against a same-data baseline |
| CS02 Precision-agriculture UAV obstacles | Existing published agriculture measurements and **separate** genuine VisDrone aerial transfer records | Agricultural-field annotated photographs/video, a matching agricultural obstacle detector, field-class/occlusion/illumination benchmarks and real observed detections; VisDrone pedestrians/cars are not automatically agricultural obstacles |
| CS03 YOLO autonomous drone | Recorded YOLO11n on 24 genuine VisDrone images; local macro AP50 .1406456330 on 972 original objects; source integrity checked; non-bar measured dot plot | Independent held-out/temporal video evaluation; IDF1/HOTA/ID switches, speed/latency measured, controller action traces, robustness and safety-bounded vehicle operation |
| CS04 Cross-dataset driving protocol | Existing generated-data code illustrates protocol mechanics | Real paired driving datasets with comparable labels and non-overlapping held-out scenes, actual named perception-model predictions, matched baselines, domain shift/error/uncertainty plots |
| CS05 Production autonomous-driving stack | Latency/telemetry parsing implementation and fixture-level tests | Authentic instrumented autonomy-stack traces (scene timestamps, sensors, perception, tracker, planner, actuators), measured latency/error and valid control outcomes; fictional traces are disallowed |
| CS06 Multimodal driving | Real Middlebury stereo-image baseline, not fusion model | Authentic synchronized camera/LiDAR data with calibration and matched frames, pretrained or trained fusion model, real joint outputs, matched image-only/LiDAR-only comparisons and target-domain validity |
| CS07 Accelerated MRI reconstruction | A real-fastMRI-capable classical reconstruction script; no actual scan evaluated in current evidence | Access original raw fastMRI HDF5, matched trained VarNet/Adaptive-CS-Net checkpoint, subject-disjoint split, model/baseline reconstruction outputs, SSIM/PSNR plus spatial residuals, and correct scanner/acquisition limitations |
| CS08 NeRF versus 3DGS | Runner for two named Nerfstudio methods and a separate real-photo baseline | Original posed multi-view scene with an image-level held-out view set; executed Nerfacto and Splatfacto training; matched PSNR/SSIM/LPIPS, GPU memory, wall time and actual render FPS; acknowledge method identity versus historical Mip-NeRF360 |
| CS09 Zero-shot SAM segmentation | Original Meta ViT-B SAM run: 42 distinct COCO photos, 147 masks; image-cluster intervals; original matched 99-object GrabCut comparison; Mask R-CNN predictions on same 18 scenes; real-data overlay and non-bar uncertainty figures | Representative original small/crowded images, independent evaluation and training-leakage considerations, fair comparator or explicit asymmetric input-information analysis, prompt-robustness experiments and target-domain latency/transfer. Oracle GT boxes cannot substantiate prompt-free performance |
| CS10 TuSimple highway lanes | ONNX evaluator with explicit logits/probability semantics and corrected missing-label rasterization; unit tests | Genuine TuSimple original samples/labels and matching lane-model checkpoint, real predictions and lane overlays, official lane-specific evaluator and error analysis; a four-pixel raster IoU proxy is not official TuSimple scoring |
| CS11 Pedestrians under occlusion | Official pretrained Faster R-CNN on all 170 original Penn–Fudan photos, 423 annotated persons, local AP50 .9681834843 and actual overlays; CrowdHuman evaluator and measured-only plotter ready | Original CrowdHuman/CityPersons occlusion labels, actual occlusion-stratified model predictions and fair baseline, official/qualified AP or miss-rate analysis, image-cluster interval and crowded-scene failure figures. Penn–Fudan general pedestrian AP alone cannot validate occlusion robustness |

## Why the unresolved requirements cannot presently be marked PASS

1. **Empirical-input absence:** Original KITTI, TuSimple, CrowdHuman, fastMRI, and matched multi-view data/checkpoints are not present in the current local execution workspace. A public download page or runnable preflight does not establish a successful original-data inference run. Some sources involve licenses, access agreements, large downloads or CUDA-dependent model execution. These are legitimate *current environment / data access* barriers, **not proof of permanent impossibility**.
2. **Task-scope mismatch:** General Penn–Fudan detection is not an occlusion-stratified validation. Original COCO SAM with oracle boxes cannot establish automatic zero-prompt segmentation. Aerial image detection does not establish drone video tracking/control. Stereo imaging is not multimodal LiDAR fusion. These scientific category errors cannot be repaired by caption edits.
3. **Missing observational evidence:** Existing generated-data or geometry-only test fixtures in CS04/05/07 are valid for software checking but not empirical application outputs. A model diagram, published-data replot or script existence does not satisfy R3, R4, R7 or R8.
4. **Fair protocol and partition deficiencies:** Without disjoint evaluation splits, baseline parity, correct official evaluation implementation where claimed, uncertainty and failure analysis, R2/R5/R6 must remain partial or failed. CS09's 18/24 samples are disjoint *within the evaluated validation subset*, not an independent held-out domain benchmark.
5. **Manuscript verification:** The latest complete Chapter 5 compiled PDF/TeX and all assets are not present in the current local workspace. Existing repository reports describe a prior 108-page release, but this pass cannot independently inspect every final placement/caption. A figure approved on its own is not a certified integrated chapter figure.

## Non-negotiable evidence promotion

Each claimed promotion from PARTIAL/FAIL to PASS must attach: original data identifiers/split and counts; license and hashes; model configuration/checkpoint/source SHA; actual executed command/run identifier; preserved per-scene outputs; target-appropriate metrics and at least one matched baseline; uncertainty/failure analysis; task-specific validation; measured-only original-figure and underlying plotted records; precise manuscript figure number and rendered-page review; and a scientifically scoped conclusion. Any missing R keeps its unit uncertified.

No unit is certified by scripts, CI fixtures, status labels, or replotting previously recorded aggregates alone. No requirement has been waived, and there is no unsupported claim that all eleven can be completed without original data, GPU/software dependencies or appropriate domain observations.

## Next experimental execution sequence

1. Original CrowdHuman/CityPersons occlusion experiment and TuSimple lane experiment; their evaluator/provenance scaffolds exist, but input data and compatible pretrained weights must be acquired first.
2. KITTI PointPillars and synchronized original camera–LiDAR fusion on real driving scenes.
3. Autonomous drone tracked-video inference and instrumented controller traces, then cross-dataset driving evaluation and production-stack traces.
4. Agricultural-field obstacle detection, original fastMRI trained reconstruction, matched NeRF-versus-3DGS genuine GPU training.
5. Representative SAM prompt robustness and fair comparator evaluation, plus full final-manuscript figure audit. Keep non-bar, premium-palette figures scientifically matched to data.

**Certification conclusion: 0/11 FULL PASS. HOLD until all ten are supported for each unit.**
