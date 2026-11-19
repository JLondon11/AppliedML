# Chapter 5 — publication caption candidates and original source mapping for four real-model figure plates

**Production status:** `STAGED_FIGURE_ASSETS_PASSED`; **NOT** inserted into or numbered within a verified latest Chapter 5 manuscript. Actual chapter TeX/PDF must be inspected before captions, page location, displayed figure size and final production approval can be certified.

## Original stereo input, disparity and predictions (CV5-D03)

**Candidate file:** `middlebury_stereo_source_and_actual_predictions.pdf`. **Genuine output CI:** [run 44430400388](https://github.com/JLondon11/AppliedML/actions/runs/44430400388), artifact `chapter5-middlebury-stereo-actual-level-a-results`. Actual original algorithm and figure code `figure_generation/real_middlebury_stereo/real_middlebury_stereo_certification.py` and `figure_generation/real_middlebury_stereo/compact_print_stereo.py`.

**Caption candidate:** *Authentic rectified Middlebury motorcycle stereo photographs, original reference disparity, and actual stereo-disparity fields computed using OpenCV block matching (StereoBM) and semi-global block matching (StereoSGBM). Both algorithms use the same 96-pixel disparity cap and original source pair; the disparity images have a shared 0–90-pixel display scale. White regions indicate invalid estimated disparity or unavailable reference samples. This single photographic scene does not evaluate a calibrated LiDAR sensor, multimodal fusion, or broad driving-scene generalization.*

## Original measured stereo error and missing-estimate maps (CV5-D03)

**Candidate file:** `middlebury_stereo_observed_error_and_missing.pdf`. **Actual same CI:** run 44430400388. **Original valid-reference pixels:** 343,274. StereoBM valid-prediction coverage 74.0895%, SGBM 82.9434%; Bad-3 across all reference-valid pixels, assigning missing predictions as failures, BM 30.4008%, SGBM 21.8764%. The reported metrics are source-original and independently re-executed.

**Caption candidate:** *Observed spatial absolute-disparity error and prediction-coverage masks for StereoBM and StereoSGBM on the same genuine Middlebury motorcycle stereo pair. Error panels exclude pixels lacking a valid original reference and include only positive valid model disparity predictions; separate coverage panels distinguish valid estimates from missing ones. Absolute-error display ranges from 0 to 12 disparity pixels. On 343,274 original reference-valid pixels, StereoSGBM yielded greater prediction coverage (82.94% versus 74.09%) and lower missing-aware Bad-3 error (21.88% versus 30.40%) than StereoBM in this single-scene test. Source pixels are spatially correlated and do not constitute independent scenes.*

## Original pedestrian photo/annotation versus actual Faster R-CNN predictions (CV5-D02)

**Candidate files:** `pennfudan_original_detection_gallery_1.pdf` and `pennfudan_original_detection_gallery_2.pdf`. **Genuine CI:** [run 44431385219](https://github.com/JLondon11/AppliedML/actions/runs/44431385219), artifact `chapter5-level-a-original-detection-audit`. **Original evidence:** successful frozen pretrained Faster R-CNN PennFudan run 41980566831, original artifact 13393014885, original complete ZIP SHA256 `1bc0a2e7690c47c4a51a7fd1b09603134d7690cfb78bccab9dae22cb7f353fb5`; code `case_studies/occluded_pedestrian_detection/print_original_pennfudan_gallery.py`.

**Caption candidate, first page (original photos 1–3):** *Actual pedestrian annotations and detector predictions on three genuine Penn–Fudan photographed test scenes, showing the original source person annotations at left (teal outlines) and the corresponding actual pretrained Faster R-CNN detections at right (orange outlines, with predicted scores). These are source-executed model outputs, not generated or illustrative photographs. The study evaluates generic pedestrian detection; original occlusion-degree labels were not available and occlusion robustness cannot be inferred from these panels.*

**Caption candidate, second page (original photos 4–6):** *Additional genuine Penn–Fudan image pairs illustrating source-annotated person bounding boxes and actual model-detected persons for the same original photographs. The six panels are presented in original numbered source order rather than cherry-picked by prediction success. Across all 170 distinct original photographs, the original evaluator recorded 423 annotated persons, 1,199 retained detector predictions and 417 true-positive matched persons. The reported original local AP50 (0.96818) was computed by the original executed evaluator; the current result archive omits the underlying original source masks required to independently recalculate its AP. No occlusion-stratified result is claimed.*

## Publication/reuse and scientific QA requirements

Each of the four genuine original-data figure plates has been generated and inspected in a **468-point (6.5-inch)-wide source PDF** with **at least 8.2 pt vector text and no off-page labels**. Original photographs are third-party original source data; legal/licensing rights for public commercial-book reproduction must be reviewed independently. Complete source code is committed; generated photographic binaries are included in the scientific evidence artifacts rather than silently presented as licensed public repo assets.

**Unresolved:** no current exact Chapter 5 manuscript figure numbering, page-level source mapping, PDF/TeX integration, or whole-chapter publication audit. **Parent CS06 camera–LiDAR fusion** and **CS11 occlusion-specific pedestrian detection** remain `SCIENTIFIC_HOLD`. Genuine stereo and generic detection are certified only under **separate Level A demonstration** identifiers D03 and D02. All original chapter production statuses remain `PRODUCTION_HOLD`.
