# Chapter 6 — Pass 53: authentic Middlebury within-scene stereo-view synthesis certificate

**Decision: CH6-D06 SCIENTIFIC_CERTIFIED, Level A, strictly within one real original stereo scene.** This does NOT certify the chapter's broader synthetic autonomous driving or road-scene performance claims. **PRODUCTION_HOLD** remains.

## Genuine original data and fresh training execution

Original real rectified Middlebury 2014 Motorcycle left/right camera images loaded locally from `skimage.data.stereo_motorcycle()`; shape 500×741 RGB each. Genuine SHA256 of left original `ca829467c1d4f427da9c4862ba43829da6ac90afe1f75735e95dba9e3fd9620b`, right original `ae44d83f55e66623c7985499fd2f1685a56023e442e66eca89b3457dd46b17af`.

A fresh real-data conditional VAE was executed locally for **220 actual Adam training steps** (seed 20261110). An encoder receives both real left/right training images; a conditional decoder receives left image and eight latent coordinates. At evaluation, the right-camera pixels are **not available** to the inference decoder. Eight latent prior samples per held-out left patch are averaged for each right-view prediction.

Spatial original-image row regions yield **855 training / 180 validation / 225 held-out test 32×32 image patches**; each region is disjoint from the others, but the test patches belong to **the same original parent scene** and are spatially correlated. The selected checkpoint was step **220** (last step), so no convergence claim is supported.

## Real held-out measured results (newly executed)

| Original right-camera target metric | Copy-left original-image reference | CVAE generated right-view predictive mean |
|---|---:|---:|
| Mean PSNR (dB) | **23.157011094530336** | 14.408218237789152 |
| Mean SSIM | 0.5400137761007581 | **0.6198464148243268** |

CVAE beats the copy-left reference in **46/225** original held-out patches on PSNR, and **188/225** on SSIM. Mean generated-sample per-pixel variation summary: `0.12815736108356052`. This is a deliberately negative / mixed finding: CVAE images are observably blurred, the PSNR deficit is substantial, and there is no evidence of high-fidelity generalization.

## Figure/record audit

Actual generated publication-style evidence was visually inspected in the execution environment:
- `paired_metrics.png` and `paired_metrics.svg`: two-panel paired PSNR/SSIM *measured on 225 original held-out right-camera photographs*; muted teal scatter points and a true equality reference.
- `real_stereo_gallery.png`: six original spatially held-out left/right image crops, actual trained-model predictions, and per-pixel error. The figure explicitly shows poor sharpness rather than concealing it.
- `test_metrics.csv`: every one of the 225 genuine original-target patch comparisons, `training.csv`: 22 logged validation/training checkpoints, `manifest.json`: actual source hashes, source count, splits, model metrics and seed, `conditional_stereo_vae.pt`: newly trained checkpoint.

The original complete model implementation and two publication-figure generators are already committed as `code/section_6_real_middlebury_stereo_view/run.py`; the source/figure/checkpoint replay gate was added as `code/section_6_real_middlebury_stereo_view/replay_scoped_middlebury.py`. The separately generated local experimental artifact package is available from the chat output; model checkpoint files are not committed as repository source code. Exact current manuscript figure placement remains separately unchecked.

## Proportional R1–R10 determination

| Requirement | Verdict | Evidence |
|---|---|---|
| R1 authentic dataset and provenance | PASS | Original Middlebury calibrated photographs and individual source-image hashes |
| R2 partitions/leakage | PASS (within-scene) | Spatial 855/180/225 disjoint row-region patches; one parent pair disclosed |
| R3 executed named algorithm | PASS | Actually trained 220-step original-image conditional convolutional VAE |
| R4 actual generated outputs | PASS | Eight latent-prior samples and true 225-patch generative-image predictions, original-gallery evidence |
| R5 credible baseline | PASS | Unaltered original left-view photographs as direct image-translation reference |
| R6 metrics/errors | PASS | 225 paired PSNR/SSIM evaluations and per-patch comparisons; visible reconstruction failures |
| R7 domain matched | PASS (within-scene only) | Real camera-view synthesis on one authentic rectified stereo scene; no multi-scene generalization claimed |
| R8 scientific figures | PASS for scoped experiment | Data-derived quantitative paired scatter and actual original/decoded/error gallery inspected |
| R9 executable code and provenance | PASS | Full run script and reproducibility gate in GitHub, output record identities and full execution environment |
| R10 conclusions scoped | PASS | Within-scene photographic generative translation only; explicitly not synthetic driving benchmark |

**Scope restriction:** no independent scene, no autonomous driving, no downstream detector training or mAP, no photorealistic generated data quality claim. A Level A certificate for this real-data educational demonstration cannot be transferred to a broader parent driving application or imply road-safety generalization.

## Counting and QA

Pass 53 increases the **scoped** count to **six** certificates: **five Level A plus one Level B**. This is **not** six fully certified parent applications out of fourteen. Full current compiled Chapter 6 production approval remains HOLD pending source/figure integration and editorial inspection. The original-data driving BDD100K benchmark and prospective concrete laboratory outcomes remain on scientific HOLD.
