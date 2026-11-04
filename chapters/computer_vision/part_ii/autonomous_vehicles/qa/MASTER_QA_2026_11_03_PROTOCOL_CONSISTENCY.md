# Chapter 5 — Protocol consistency remediation (2026-11-03)

**Chapter 5 = Computer Vision Part II: Autonomous Systems.** Manuscript compiled in two successful LaTeX passes as a 98-page PDF, with 45 figure environments, 12 tables, and no unresolved cross references/citation warnings or overfull hboxes in the final build.

## Corrected manuscript evidence rules

- SAM: SA-1B pretraining cannot be confused with independent COCO validation; the named checkpoint, oracle ground-truth box prompt, per-instance mask IoU/Dice and GrabCut paired comparison are separate from prompt-free / official COCO AP. Numbered equation `eq:cv5_sam_mask_overlap`.
- TuSimple: local rasterized mask IoU is not official TuSimple lane-point accuracy. Numbered equation `eq:cv5_lane_raster_iou`; record line width, decoder type and annotation split.
- CrowdHuman: full-body matching and visible/full-body bounding-box area proxy are distinct from official CityPersons occlusion categories and CrowdHuman AP/MR. Numbered equation `eq:cv5_pedestrian_occlusion_proxy`.
- Production driving stack: per-frame summed stage latency and deadline miss fraction require **genuine recorded traces**; asynchronous pipelines need measured end-to-end timestamps, and emergency braking commands are not collision/safety outcomes. Numbered equation `eq:cv5_production_latency_deadline`.
- Replaced long raw GitHub URLs with readable section-specific hyperlinks.

## Verified artifacts

Local release package `Chapter_5_Protocol_Master_QA_20261103.zip` contains the 98-page compiled PDF, full .tex manuscript, all available referenced figure assets, available Python implementations and an evidence report. Python syntax checks passed on the bundled case-study scripts. The actual source manuscript update is in the downloadable package and is **not claimed to have been committed to GitHub**.

## Blocking experiments

All eleven named studies remain unapproved until model-specific original-dataset inference/training, quantitative tables, generated model-result figures, primary dataset citations, checkpoint provenance, matched benchmark comparisons, and scientific analysis have been individually verified. Classical real-image stereo, generated-scene tracking, phantom MRI, and published-measurement replots cannot substitute.

**Overall: manuscript protocol QA improved; Master QA NOT APPROVED.**
