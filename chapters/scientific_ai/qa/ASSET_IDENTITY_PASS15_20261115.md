# Chapter 11 — PDF embedded-image asset identity audit, Pass 15

## Reconciled source and scope

- Current **177-page** scientific AI PDF SHA-256: `49073b974df666930e2b32a339e3e4d7b5daf297f4835abc2ad224282ed98b0a`.
- Earlier editable source and figure library: `All_Chapters_Complete_10.zip`, 54 historical figure images.
- Captions and *actual embedded PDF image objects* were associated by bounding boxes on the same printed PDF page, not by filenames or prose mentions.
- Matching method: 64-bit DCT perceptual hash plus normalized pixel correlation on rescaled grayscale images. High-confidence requires Hamming distance ≤3 and correlation ≥.985.

## Outcome

- **68/68** current numbered figures were mapped to an embedded PDF image.
- **52/68** were verified as content-identical to historical image files (pHash distance **0**, normalized pixel correlation **1.0**).
- **16/68** had no matching historical image. All **16** were recovered from embedded PDF content as PNGs into a local Pass 15 evidence ZIP with per-file SHA-256 manifest. **Those binaries were not committed by this QA record.** They are preserved artwork, not proof of original dataset/model reproducibility.

## Critical confirmed identity collision

- Current PDF **Figure 11.54**, PDF page **139**, is the *Domain-Scale Scientific AI Platform* diagram, content-identical to historical `Figure_11_41_Domain_Scale_Scientific_AI_Platform.png`.
- Current PDF **Figure 11.68**, PDF page **165**, is the *Multiscale Sensor and Observation Assimilation Architecture*, content-identical to historical `Figure_11_53_Bayesian_Multiscale_Digital_Twin_Assimilation.png`.
- Neither is the measured heated-rod results figure in `chapters/scientific_ai/code/figure_11_54_real_rod_digital_twin`. The independent lab experiment evidence **does not establish integration** into this particular current PDF.
- Revised **Figures 11.3 and 11.4** are among the 16 figures without matches in the historical image library.

## Separate publication decisions

Scientific chapter certification: **HOLD**, due to incomplete named-task experiments, notably paired annotated nucleus instance segmentation/classification, sepsis, retinal disease classification, volumetric CT, and materials discovery.

Chapter production approval: **HOLD**, due to missing exact current-edition matching editable LaTeX source, complete per-figure experimental lineage, and GitHub canonical release. The recovered binaries and 52 old matches can seed a **traceable reconstruction**, but are not a replacement for a verified source/asset release.

Local deliverables: `Chapter11_Asset_Identity_Pass15.zip` containing `pdf_embedded_asset_crosswalk.csv`, `recovered_asset_manifest.csv`, `asset_verification_summary.json`, `recovered_current_pdf_figures/`, `verify_embedded_assets.py`, and `extract_unmatched_assets.py`.

These findings update, without rewriting, the historical Pass 14 record. Follow `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md` and `CODE_PUBLICATION_STANDARD.md`.
