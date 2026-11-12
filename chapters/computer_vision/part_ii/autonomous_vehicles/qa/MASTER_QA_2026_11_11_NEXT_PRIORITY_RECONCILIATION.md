# Chapter 5 — next-priority evidence reconciliation (2026-11-11)

**Chapter identity:** Chapter 5, Computer Vision Part II — Autonomous Systems. **Decision:** HOLD. This is a source-tree and existing-results audit, **not** a new model execution or manuscript certification.

## Evidence reviewed

- `CH5_CODE_MANIFEST.csv` (11 named case-study/application units, all marked BLOCKED)
- `qa/MASTER_QA_2026_11_05_SAM_42_AND_MASKRCNN.md`
- `qa/MASTER_QA_2026_11_04_EXPANDED_REAL_VALIDATION.md`
- GitHub default-branch recursive tree for this chapter
- `qa/MASTER_QA_STATUS_AND_PRIORITIES_20261110.md`

## Verified status distinctions

| Unit | Evidence already reported | Next decisive experiment / remediation |
|---|---|---|
| CS01 KITTI 3D | PointPillars runner, renderer, tests exist; no verified KITTI inference | Acquire licensed original KITTI split and compatible pretrained checkpoint; execute 3D inference, report official AP and latency with measured visualizations |
| CS02 agricultural UAV obstacles | Published-data replot and VisDrone integrations; not agricultural-field model validation | Execute named agricultural-field detector on original annotated field/UAV data; measure precision/recall by obstacle and scene, show genuine detections |
| CS03 YOLO drone | Actual 24-image VisDrone transfer result; not UAV tracking/control | Run original video sequences for tracking IDF1/HOTA and trajectory errors; evaluate planning/control separately on observed/simulation traces with labels |
| CS04 cross-dataset driving | Existing generated-data experiment is not a substitute for original scene perception benchmark | Train/validate/test on documented real driving datasets; quantify cross-domain degradation and uncertainty; no synthetic result figures as empirical evidence |
| CS05 production driving stack | Telemetry evaluator fixture-tested, no authentic end-to-end traces | Obtain measured real-stack traces with timestamps, predictions, perception/planning latency and failures; produce measured bottleneck/error figures |
| CS06 multimodal driving | Middlebury stereo photographs baseline, not camera–LiDAR fusion | Execute registered original synchronized camera/LiDAR pairs with actual fusion inference and image/BEV error visualization |
| CS07 accelerated MRI | fastMRI-capable code, no verified actual fastMRI model results | Execute genuine raw fastMRI HDF5 and trained VarNet/Adaptive-CS-Net; declare slice/volume/patient splits and report PSNR/SSIM, artifact maps |
| CS08 NeRF versus 3DGS | Paired GPU training integration; no matched neural-rendering run | Fit both methods on identical genuine posed multi-view scene; report held-out PSNR/SSIM/LPIPS, throughput, memory, visual comparisons |
| CS09 SAM | Real COCO 42 photographs / 147 original instances; measured SAM and GrabCut; 18-image same-scene Mask R-CNN; output figures reported in chapter archive | Broaden to representative COCO and driving images, prompt-robustness, fair information-budget comparisons, official AP where applicable; don't interpret oracle-prompt SAM vs automatic Mask R-CNN as equal input |
| CS10 lanes | TuSimple ONNX evaluator and tests; no actual named-model run | Run compatible pretrained lane model on original TuSimple split and labels; calculate official metrics and show genuine highway masks/lanes |
| CS11 occluded pedestrians | Real all-170 Penn–Fudan Faster R-CNN baseline with genuine predictions, not an occlusion-specific test | Run original CrowdHuman/CityPersons occlusion annotations and stratify AP/miss rate by visibility and crowding |

## Evidence-sensitive release decisions

1. **Cannot certify any of the 11 full units from the currently inspected manifest.** This does not erase genuine partial results from CS03, CS09 and CS11.
2. For each unit, require: original data source/license, exact image/scene/patient counts and train/validation/test protocol, checkpoint and code SHA, recorded full inference outputs, measured metrics plus uncertainty/failure analysis where appropriate, scientific figure assets and captions, manuscript section integration, and visible PDF review.
3. Figure quality must be assessed for scientific relevance, real-data lineage, legibility, premium restrained color palette, balanced plot types, and correct pixel-to-caption correspondence. An explanatory schematic is not an experimental output.
4. **Manifest repair necessary:** CS11 presently says `named_model_real_dataset_inference=NO` despite a verified Penn–Fudan pretrained Faster R-CNN run; that's defensible only if the field means *named occlusion-specific model*, which its header does not communicate. CS02 also predates newer VisDrone evidence. Preserve partial factual evidence in descriptions and do not flip a unit to certified.
5. The actual latest Chapter 5 LaTeX/PDF assets were **not** present in the current local workspace. Existing GitHub QA describes a 108-page release, but visual manuscript validation and new PDF integration cannot be independently repeated in this pass.
6. No additional trained-model result or new empirical figure was produced by this audit. Do not convert runnable integrations or passing fixture tests into measured experiments.

## Execution order

**First:** CS10 TuSimple and CS11 occlusion-specific evaluation, because they have evaluators and direct quantitative visual targets. **Second:** CS01 KITTI inference and CS06 camera–LiDAR fusion, required for the chapter's core autonomous-perception coverage. **Third:** CS03 original video tracking and CS04 real cross-dataset evaluation. Then resolve CS02 agricultural domain, CS07 fastMRI, CS08 NeRF/3DGS, and CS05 production telemetry, conditional on access to actual datasets/checkpoints/GPU.

**Master QA decision:** HOLD until named-model evaluations run, output plots are linked to original predictions, all 11 unit claims match their titled scientific scope, and the integrated PDF is independently reviewed.
