# Chapter 6 Pass 47 — additional numerical verification

**Full Master QA: HOLD — 0/14 fully certified active case studies/applications.**
The Pass 45 (109-page) manuscript remains the latest compiled chapter. Pass 47 is an independent saved-evidence verification addendum, not a new edition and not independent retraining of all models.

## Seven new passed saved-evidence checks

1. Real tissue patch CVAE: genuine image SHA-256, 589/93/217 disjoint source regions, 217 measured-patch reconstruction records, 128 prior samples, model checkpoint.
2. Real-photograph one-step denoising: six genuine source photograph hashes, 48 actual test metric rows, six model checkpoints. Two held-out photos; not full reverse diffusion.
3. Digits latent diffusion: genuine 1,797 real digit images, 1257/270/270 split, 2,000 saved predicted labels/probabilities, output figures. Not pretrained natural language Stable Diffusion.
4. NCI SMILES VAE: 600 genuine NCI molecular structures, 120 held-out test rows and 1,024 generated prior strings, trained model evidence. Extremely low unconditional validity (~3.52%), held-out exact reconstruction 0%.
5. NCI molecular search: 4,886 real NCI source structures, 978 held-out computed descriptor predictions and 300 generated edited molecular structures. No new activity assay.
6. Wisconsin clinical cytology: genuine data checksum, 341/114/114 genuine cases split, all 1,026 saved real held-out classifier predictions rechecked for nine training-arm/seed scores, 1,023 generated tabular observations. Mean ROC AUC real-only 0.99934, CVAE augmented 0.98071; negative augmentation result.
7. Mauna Loa CO2 forecasting: source checksum, 1,145/254/306 chronological windows, all 7,344 saved forecast-horizon predictions, 12 reported RMSE/MAE scores and central 90% predictive interval coverage. Actual coverage averages ~20.25% over 3 seeds, a failed uncertainty-calibration test.

The earlier Pass 46 independently verified the real NEU benchmark, cross-domain probabilities, single-scene Middlebury results and all 21 Brownlee industrial evaluation folds. Concrete figure inputs received a partial check but the 600-candidate generator was not independently rerun here; no generated recipe has been physically cast.

## Figures

Two new supplementary scientific figures were generated directly from saved held-out real-data results: a lead-resolved CO2 coverage/RMSE comparison and a real-test-cohort cytology ROC/seeds comparison. These are **not yet integrated** in the Pass 45 manuscript and cannot be counted as a new chapter edition. The same 306 rolling forecast windows and 114 clinical test cases recur across seeds; do not treat them as independent observations.

## Unresolved requirements

- Actual BDD100K multi-scene model training and held-out full-scene downstream detection mAP. Software fixture tests alone cannot count.
- Signed, independently measured 28-day concrete cylinder strengths for newly generated recipes. Surrogate scores are not physical results.
- Out-of-parent/subject/scene generalization where the parent source pool is only one microscope image, three texture images, one stereo pair or one industrial process.
- Clinical privacy/utility and external patients, calibrated probabilistic CO2 uncertainty, molecular target and wet-lab validity, independent clean-checkout retraining and run-to-figure tracing.

Files delivered to the user: Chapter6_MasterQA_Scientific_Evidence_Pass47.zip; Chapter6_MasterQA_Certification_Matrix_Pass47.csv; Chapter6_MasterQA_Scientific_QA_Pass47.md. The archive contains the executed verifier, saved source evidence (no original BDD100K driving data), publication-ready supplementary PDFs/PNGs, and audit matrix.

**Do not upgrade any full-unit certification until the task-specific gates above have actual results.**
