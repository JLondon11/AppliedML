# Chapter 5 — original Middlebury real-data StereoBM versus StereoSGBM scientific Level A certification

**Date:** 2026-11-12. **Decision:** **CV5-D03 = Level A SCIENTIFIC_CERTIFIED (R1–R10 PASS)** for the explicitly titled, **single-scene classical rectified stereo disparity-estimation demonstration**. **Production HOLD** for final chapter PDF placement. **Parent CS06 multimodal camera–LiDAR autonomous-driving application remains SCIENTIFIC_HOLD**, because the stereo experiment has no LiDAR input or fusion model.

## What was actually executed

- The genuine original Middlebury motorcycle left/right rectified photographs and supplied reference disparity were loaded via `skimage.data.stereo_motorcycle()` (741×500 RGB image pair, original reference disparity).
- The scene has **343,274 valid reference pixels** after requiring finite GT disparity >0 and <96 pixels.
- Actual native OpenCV **StereoBM** and **StereoSGBM** algorithms executed with the same disparity limit (96) on the same original images. BM uses blockSize 15; SGBM uses blockSize 7, three-way aggregation and its documented smoothness/speckle parameters. Original pixel-byte SHA-256 values are pinned in `figure_generation/real_middlebury_stereo/recorded_provenance_20261112.json`.
- No training or fine-tuning was performed. No fabricated train/validation/test split is needed for two classical inference algorithms evaluated on a fixed real-data demonstration.
- Original source/implementation and plot generator are committed at `figure_generation/real_middlebury_stereo/real_middlebury_stereo_certification.py`, original observed numerical metrics at `original_measurements.csv`. Model-disparity output is regenerated from the original source photos, not invented. Three original-outcome figure formats (PDF/PNG/SVG) were generated in the run and placed in the downloadable local evidence package.
- The final scientific plot was rendered from PDF and visually inspected. It contains **original photo, reference disparity, actual two predicted disparity maps, valid-reference-masked spatial error image, and empirical error CDF**; six substantive panels, no bar charts. A source-map error-mask defect was detected and corrected before scientific approval: pixels with missing original reference disparity are now excluded from the *error image*; these missing estimates remain accounted for in all-reference error metrics.

## Actual measured source results

| Metric | StereoBM | StereoSGBM |
|---|---:|---:|
| Original reference-valid pixels | 343,274 | 343,274 |
| Valid predicted pixels | 254,330 | 284,723 |
| Coverage of valid reference | 0.740895 | 0.829434 |
| Valid-prediction MAE, disparity pixels | 1.267169 | 1.222608 |
| Valid-prediction RMSE, disparity pixels | 5.480065 | 5.315820 |
| Bad-3 across all valid reference, including missing estimates | 0.304008 | 0.218764 |
| Local CPU median of three warm calls | 9.09 ms | 23.87 ms |

**Interpretation:** On this genuine fixed original image pair, SGBM predicts at a larger share of reference-valid pixels and has lower observed error by these chosen measures; its local warm CPU runtime was higher. Error metrics are on a **single scene** with spatially correlated pixels. **Do not pretend 343,274 pixels are 343,274 independent experiments or assign unsupported confidence intervals/generalization claims.** Exact warm runtime is not portable between hardware and reruns.

## Proportional Level A R1–R10 determination

| Requirement | Status and scientifically proportionate rationale |
|---|---|
| R1 original data/provenance | PASS — authentic Middlebury stereo photos + original reference disparity, exact source pixel-byte hashes and environment versions |
| R2 partition/leakage | PASS — no learned fit; both fixed classical methods process the same source pair; no artificial training splits |
| R3 executed algorithm | PASS — both named OpenCV stereo algorithms actually executed |
| R4 actual outputs | PASS — actual predicted disparity fields, spatial error image and derived recorded metrics |
| R5 credible baseline/reference | PASS — both classical methods are valid controlled comparators and the supplied original disparity is the quantitative reference |
| R6 suitable errors | PASS — MAE/RMSE, reference coverage, missing-aware all-reference Bad-3 and observed error CDF; no unsupported population CI |
| R7 domain validation | PASS — direct rectified stereo photographs and actual measured disparity reference; only offline stereo is claimed |
| R8 measured scientific figure | PASS — six-panel source-data/model-output figure, scientifically corrected reference validity masks, visually inspected at rendered size |
| R9 source/provenance traceability | PASS — complete executable implementation, source asset hashes, parameter settings, numeric metrics, and figure-generation source committed to GitHub; source image obtainable through documented genuine scikit-image dataset API |
| R10 supported scope | PASS — claims only original single-pair classical stereo; no neural rendering, LiDAR fusion, or broad benchmark generalization |

## Strict boundaries

**Do not count this Level A certification as CS06 Level B camera–LiDAR fusion certification.** CS06 still requires true calibrated synchronized camera–LiDAR observations and an executed model actually fusing both sensor modalities.

The experiment fulfills R8 with a genuine original result figure but does not establish final Chapter 5 editorial-production compliance. No latest full Chapter 5 PDF/TeX was available for in-situ page/caption review. All original dataset sample/content and qualitative figure provenance are scientific, but image copyright and redistribution are governed by the original data package license; never label the data as simulated or store protected data without permission.

## Provenance and follow-up

Scientific source: `figure_generation/real_middlebury_stereo/real_middlebury_stereo_certification.py`; source-checkpoint alternative: deterministic OpenCV BM/SGBM parameter configuration; dataset pixel identities: `recorded_provenance_20261112.json`; local numeric result record: `original_measurements.csv`. The new scoped demonstration appears as `CV5-D03` in `qa/CH5_LEVEL_A_EMBEDDED_DEMONSTRATION_CERTIFICATIONS_20261112.csv`.

**Combined current Chapter 5 accounting:** **four scientifically certified experimental units:** CS09 (Level B) plus D01 original YOLO aerial detection, D02 original Penn–Fudan pedestrian detection, and D03 original Middlebury stereo (each Level A). **The main eleven case study/application parents are still 1/11 scientifically certified**, and book production is separately HOLD.
