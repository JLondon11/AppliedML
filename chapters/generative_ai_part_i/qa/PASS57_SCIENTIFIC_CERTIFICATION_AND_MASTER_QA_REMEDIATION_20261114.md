# Chapter 6 — Pass 57 scientific certification and updated Master QA remediation

**Date:** 2026-11-14  
**Canonical repository:** `JLondon11/AppliedML`  
**Scientific rule:** proportional R1–R10 (A/B/C), real original datasets and executed model outputs; separate scientific evidence from manuscript-production approval.

## Scientifically established state (not extrapolated)
The current scoped register lists **eight** certified experiments, **seven Level A and one Level B**. These are internally scoped demonstrations, **not eight fully certified parent applications**; the archived Pass 44 parent inventory has fourteen application/case-study units. Scientific statuses in the register are certification *records*, not a replacement for reading each original model trace and artifact. The current full Chapter 6 `.tex`/`.pdf` has not been available in the accessible GitHub chapter root; production remains **HOLD**.

### Eight scoped units and critical limits
| Scoped unit | Level | Evidence boundary |
|---|---|---|
| D01 Brownlee historical process GP | A | Static retrospective emulator; ridge RMSE better than GP; not a deployed digital twin |
| D02 genuine scanned handwritten-digit VAE/PCA | A | Real 270-image test; PCA slightly lower MSE, no VAE superiority |
| D03 Mauna Loa CO₂ conditional VAE | A | Genuine held-out temporal forecast; nominal 90% intervals miscalibrated (20.25% original coverage) |
| D04 NEU steel defect conditional-GAN augmentation | B | Original real test photos; GAN harms downstream classification under tested short-training budget |
| D05 Yeh concrete strength screening | A | Authentic held-out measured specimens; proposed generated mixes are computational only and **0** physical cast/test records |
| D06 original Middlebury motorcycle stereo view | A | Only one real stereo scene and 225 correlated held-out patches; not synthetic-driving/downstream perception evidence |
| D07 Wisconsin cytology conditional VAE augmentation | A | Real held-out cytology tabular records; GAN/CVAE mathematical augmentation does not confer clinical claims |
| D08 original-photograph epsilon-prediction restoration | A | Two independent held-out real photographs; single-step denoising is **not** full DDPM diffusion sampling |

## New genuine-data NEU scientific graphic and source

Added a **new publication-ready vector SVG** at:

`code/section_8_8_2_neu_defect_benchmark/figures/neu_seed_paired_accuracy.svg`

Its **complete stand-alone generator** is:

`code/section_8_8_2_neu_defect_benchmark/plot_neu_paired_accuracy.js`

Input is the existing authentic measured file:

`code/section_8_8_2_neu_defect_benchmark/replicated_seed_metrics_pass13.csv`

The source file has Git blob SHA-1 `56de24a4f74f2c87e0d01b3b97c811e00b317f75`. The generated SVG has Git blob `d6bcd54c620755720cb3b1b8b67d79b71356f5c5`; generator source blob `3084b98b5791088ddf93fbedd3aaa1f54fe59afd`. This is original-score-derived figure provenance, **not** a claim about embedding the figure into an unavailable chapter manuscript.

The original **360 distinct real test photos** are evaluated repeatedly by three trained random seeds across four arms (12 trained classifier-arm/seed evaluations, not 4,320 independent photos). Mean genuine held-out accuracy (three seeds):
- Real-only: **0.7194444445**;
- Horizontal-flip: **0.7000000000**;
- Conditional GAN augmented: **0.5268518518**;
- Real repeats/flips: **0.7962962963**.

Measured **GAN minus real-only** differences by seed: **−18.6111, −19.4444 and −19.7222 percentage points**; mean **−19.2593 percentage points**. GAN minus real repeats/flips is **−26.9444 percentage points** on average. The figure uses continuous per-seed point/line trajectories and paired-delta dot segments instead of distracting wide bars. It does not fabricate generative-fidelity metrics or prove population-level superiority from three seeds. **The scientifically honest result is negative for the tested augmentation protocol.**

## Updated source-based automation

Modified `.github/workflows/chapter6_scoped_level_a_certification.yml` to:
- require the current **eight** scoped records (7A/1B);
- run the scoped-register and release-gate Python unit tests;
- execute `node plot_neu_paired_accuracy.js` on the authentic committed CSV; and
- fail when regenerated figure bytes differ from the committed SVG (`git diff --exit-code`).

The workflow was committed but **a completed GitHub Actions run was not independently confirmed in this pass**. Do not report a CI PASS merely from workflow configuration.

## Critical hold gates under updated Master QA
1. **Figure 6.20:** legacy digits generator has a filename collision with the reportedly industrial-defect figure; the exact current image/caption and original generator are not available to match. Prior Pass 37 user-uploaded handoff text reports placement on page 67 but contains no usable actual manuscript PDF/TeX in the active runtime; no scientific source mapping is established.
2. **Synthetic driving:** no verified genuine BDD100K multi-scene training-plus-held-out downstream full-scene perception benchmark for the broad claim. One-scene Middlebury CVAE does not replace it.
3. **Novel concrete mixtures:** the original Yeh test rows validate retrospective surrogate prediction only; independent signed new-mixture physical specimen tests remain absent. Do not present predicted strengths as measured.
4. **All 14 parent units:** require exact manuscript titles/figures, dataset lineage, complete executed application code, real predictions, relevant baselines/errors, scientific figure and appropriate proportional R1–R10 adjudication before certifying parent scope.
5. **Production:** obtain and inspect complete current PDF/TeX, exact figures, captions, narrative, figures before chapter problems, meaningful tables and code listings, unresolved references, and final asset/code traceability.

**Certification decision:** Retain eight narrowly scoped scientific certificates with explicit caveats. No newly approved *parent application* and no Chapter 6 full-release certificate. **Master QA: HOLD**, not fictitious PASS.
