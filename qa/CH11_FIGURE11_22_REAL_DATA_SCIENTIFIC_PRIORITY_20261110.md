# Chapter 11 next scientific priority — Figure 11.22 (2026-11-10)

## Actual manuscript claim audited

The integrated Chapter 11 `Scientific_AI_InProgress.tex` describes `Application: Nuclear Segmentation and Cancer-Cell Classification`. Its existing Figure 11.22 caption refers to a **nuclear image phantom**, **28 individually labeled nuclei**, and **predicted malignant/benign nucleus classes**. That source does not demonstrate original annotated MoNuSeg images or a trained nucleus-level cancer classifier behind those assertions. The existing panel is therefore not acceptable as experimental evidence for the named application.

## Executed genuine-data diagnostic

An authentic 512×512 immunohistochemistry tissue photograph distributed with `skimage.data.immunohistochemistry()` was processed with an explicit unsupervised hematoxylin stain threshold and marker-controlled watershed. The resulting **189 stain-positive candidate regions** and their pixel areas/eccentricities are actual computed image measurements. Because there are no independent ground-truth masks, **these are not certified nuclei** and no instance Dice, IoU or AJI is claimed.

A **separate, unpaired** Wisconsin Diagnostic Breast Cancer diagnosis experiment used 569 authentic patient records: 341 training, 114 validation and 114 held-out testing. The measured logistic-regression ROC AUC was **0.9957010582010581**, and held-out accuracy **0.9824561403508771**; a real model generated the held-out probability array. WDBC diagnoses cannot be assigned to the pictured stain-positive objects.

The five-panel proposed figure contains the real photograph, stain-domain candidate boundaries, candidate-region overlay, actual morphology scatter, and **separately labeled** genuine WDBC held-out ROC. The first less-stringent stain threshold produced many large gland/cytoplasmic candidates; the final conservative 95th-percentile threshold retained 189 candidates and avoids any claim of ground-truth nuclei.

## Files and reproducibility

- Executed local archive: `Chapter11_Real_Microscopy_Candidate_Segmentation_20261110.zip`, contains Python implementation, original-image-derived PNG/PDF/SVG, full numeric JSON with predictions and morphology, README, and a proposed LaTeX insertion.
- GitHub script: `chapters/scientific_ai/code/figure_11_22_real_microscopy_candidates/run.py`
- GitHub test: `chapters/scientific_ai/code/figure_11_22_real_microscopy_candidates/test_outputs.py`
- GitHub README and verified summary in the same folder.
- Proposed LaTeX in that folder preserves the original Figure 11.22 label, explicitly separates independent datasets, and removes unsupported classification claims.
- GitHub Actions workflow `.github/workflows/scientific-ai-figure11-22-real-microscopy.yml` regenerates and checks images and numerical predictions. Its successful execution has **not yet been confirmed**.

## Certification decision

**PARTIAL REAL-DATA EXPERIMENT, NAMED APPLICATION HOLD.** These observations are genuine, and all local plotted values are derived from executed code, but the microscopy analysis has no reference masks and the WDBC diagnostic labels are unrelated to the photographed cells. The original case study requires source-authenticated MoNuSeg images plus annotations, an actual trained instance segmentation model with measured AJI/Dice/panoptic metrics, and original paired nucleus-class diagnostic labels if malignant/benign classification of segmented objects is claimed. Until those are available, neither the 189 segments nor ROC score qualifies as a MoNuSeg or cancer-cell-classification outcome. The final chapter manuscript PDF has not been rebuilt with the proposed panel.

**Master QA overall remains HOLD**, and the Chapter 11 case-study count requiring complete output evidence is not reduced by this partial demonstration.
