# Chapter 6 — Pass 62: genuine single-tissue VAE paired failure and production QA

**Date: 2026-11-15.** **Outcome:** actual inference using the authentic original histology photograph and a retained, executed 12-epoch VAE checkpoint; independently computed all 217 original held-out per-patch squared errors against a target-aware nearest original training-patch reference; designed and visually inspected a scientific original-image error atlas and archived all numerical records.

**Certification accounting unchanged:** 12 real-data scoped scientific experiments listed in the Chapter 6 registry (11 A, 1 B). This extends **existing CH6-D11 Level A evidence**, it is not a thirteenth independent application. The 14 broader original Chapter 6 application/case-study titles are not silently certified. **Full manuscript production approval = HOLD.**

## New actual original-data experimental result

Original genuine source: `skimage.data.immunohistochemistry()`, 512×512 actual photographed RGB microscopy, original source-pixel SHA256 `c5b3ef509a92f16d4c29be8cf0300fe75d53e13a3ce650159db932caea8dcc1b`. Source regions: 589 training patches (rows 0–319), 93 validation patches (rows 320–383) and 217 test patches (rows 384–511). Patches within each region overlap; **there is only one source photograph, not 217 independent patients**.

Actual trained 12-epoch VAE artifact SHA256: `204d5bd26a3361584285f108c7e74b9bdad2e5ee9d52310c7a5204f12d08bbbe`. The actual 24-dimensional model was loaded from that already executed checkpoint and decoded the original 217 test patches without stochastic re-training or invented inputs.

For each test patch, compute a **nearest original training patch by minimizing squared RGB pixel difference using the true target test patch**. This target-aware oracle is an informative reconstruction-reference bound; it would require the true test image at inference and **is not a deployable or information-matched competitor**.

| Original paired outcome across 217 source patches | Observed value |
|---|---:|
| VAE reconstruction mean pixel MSE | 0.010481160134077072 |
| Target-aware nearest real training-patch mean pixel MSE | 0.010597637985416684 |
| Mean (oracle minus VAE) pixel MSE | +0.00011647771936648178 |
| Median (oracle minus VAE) pixel MSE | **−0.0004502781666815281** |
| VAE lower per-patch MSE | **92 of 217 (42.4%)** |
| Nearest training patch lower per-patch MSE | **125 of 217 (57.6%)** |
| Exact tied results | 0 |

**Critical scientific finding:** The VAE has a slightly lower **mean** error while its pixel reconstruction is worse than the target-aware oracle on **most individual observed patches**. This is consistent with uneven behavior, including blurred fine structures. Neither result supplies an independent tissue/patient population estimate; no misleading independent-patch confidence interval is given.

## Seventh genuine non-plot figure

New scientific original-image gallery `actual_tissue_paired_failure_success_atlas.pdf/.png/.svg` shows eight actual held-out original tissue patches side by side with real trained-VAE reconstructions, their actual oracle-nearest original training photograph patches, and spatial absolute RGB error. Four cases are the **largest VAE improvements** and four are the **largest VAE degradations** on the 217 original patch pairs; case selection is explicitly deliberate, not representative random sampling. This is a physically meaningful model-output gallery, **not a bar chart or decorative schematic**.

- Figure generator/source: `code/section_6_medical_imaging_histology_vae/audit_true_tissue_vae_vs_nearest.py`, committed and callable with `--checkpoint <real .pt> --output <directory>`. It performs original model inference, computes the oracle source-patch reference, and exports **all 217 original paired measurement rows** in `actual_217_patch_paired_errors.csv`, original summary JSON and vector/bitmap gallery.
- Contemporaneously generated and visually inspected **vector PDF SHA256:** `a01e0f2a0f3273c3b1415d5ea6961150bfb41b8c6df74788db1769d49970cdfd`. PDF examined using PyMuPDF and raster preview: **one page, 32 genuine source/model/error image panels, zero off-page text spans**.
- Explicit original numerical checkpoint-identity record committed at `code/section_6_medical_imaging_histology_vae/qa/pass62_original_paired_tissue_measurements.json`. All 217 original rows and binary model checkpoint are retained in user downloadable Pass62 evidence package; because image/checkpoint binaries are not deposited in the GitHub source tree, **do not claim they are versioned in the GitHub repository**. The code is complete and committed; CI independently re-runs from genuine source and its resulting checkpoint.
- `qa/verify_pass62_paired_measured_records.py` independently checks all original held-out indices/positions, the model and real-image reference mean errors, all 217 paired signs and the mandatory target-aware-oracle limitation. Local source-equivalent execution of the 217-row validator **PASSED**. The committed QA validator distinguishes exact original-checkpoint mode from `--independent-rerun` mode because retrained model outcomes can vary across numerical environments.
- The new experimental source, complete observation-record verifier and source-photo VAE re-execution have been added to `.github/workflows/chapter6_image_first_authentic_figures.yml`. **Remote GitHub Actions execution is not yet independently confirmed** and must not be recorded as CI PASS.

## What production work was accomplished

The original Pass61 package of six source/photo/model/architecture scientific figure PDFs was directly extracted and its bundled figure preflight re-executed, with **6 original PDFs passing SHA and one-page geometry integrity**. The new seventh actual-model-output scientific PDF was rendered, inspected, and checked for page clipping. The portfolio now contains **three authentic input-source image galleries, two model architecture schematics, and two actual trained-model output galleries**, rather than being dominated by simple plots. The original input galleries are clearly not misrepresented as trained predictions; the architecture figures are labelled conceptual; the two genuine model galleries use authentic source photographs and checkpoint outputs.

A staging ledger entry (`quality_assurance/pass59_image_first_figure_staging.json`) includes source code, dataset and checkpoint SHA256 and the seventh PDF SHA256. **Actual manuscript Figure 6.xx numbers are intentionally not assigned.** Chapter 6 Figure 6.20 continues to have a mismatched provisional handwritten-digit source candidate against an industrial-defect manuscript domain and is not silently remapped.

### Scientifically appropriate suggested caption

*Paired reconstruction failures and successes of the trained tissue-patch variational autoencoder on genuine spatially held-out patches from one immunohistochemistry photograph. Eight original tissue crops are compared with their VAE reconstructions, target-aware nearest authentic training-patch references and absolute RGB reconstruction error. The target-aware reference is selected using held-out true target pixels and is not an operational model comparator. Of all 217 patches, the VAE yields a lower reconstruction MSE for 92 and the reference for 125; the VAE nevertheless has a slightly lower mean MSE. The displayed patches are selected largest improvements/degradations rather than randomly sampled examples; independent tissue/patient performance is untested.*

## Remaining scientific and production requirements

1. The current latest full chapter PDF/TeX manuscript is not in the accessible connected GitHub tree or active workspace; only a prior text handoff documents earlier Figure 6.20 placement.
2. Full 48-figure / 17-table / 11-code-listing / 14-parent scientific and editorial production inventory is not available for actual inspection; complete parent-application certification is not inferred from 12 narrower experiments.
3. Broad synthetic-driving claims still require authentic multi-scene driving generation with downstream model evaluation; physical performance claims for proposed new concrete mixes require real specimen tests.
4. No trained-model or figure step should be described as run on an independent hospital cohort, and no target-aware oracle should be compared as though deployed with identical information.

**Pass 62 decision: actual new original-data scientific analysis and original-image figure complete; numerical integrity locally verified; source and continuous-check code committed; six prior vector figures rechecked; seven staged non-plot figures verified. Full Chapter 6 scientific-parent/production approval remains HOLD.**
