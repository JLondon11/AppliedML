# Chapter 6 — Pass 50: real NEU steel-defect GAN-augmentation Level B certificate

**Decision: SCIENTIFIC_CERTIFIED for CH6-D04 — a precisely scoped offline six-class steel-defect classification augmentation experiment using authentic original NEU photographs.** The broader application of generative synthetic-defect image quality, reliable industrial inspection deployment, cross-factory generalization or real-time defect monitoring **is not certified**. Production approval of the latest full chapter PDF remains HOLD.

## Evidence and original samples

The actual original NEU-DET experiment in `code/section_8_8_2_neu_defect_benchmark/` was executed and independently audited in the preceding archived Pass 13 and Pass 46 reports, not inferred from software stubs. Dataset source archive SHA256 `2ccaff85d88567c52708c2f9dc4dec10cff8650749df2984f03b38716998460f`; **1,800** original grayscale steel-surface-defect photographs, 300 per original class. **One original exact duplicated image** within the development portion was excluded; evaluated unique source split: **1,080 real training / 359 validation / 360 held-out test** photographs, with proper source-file and decoded-pixel SHA256 inventories.

Exactly **three original training seeds** (`20261106`, `20261107`, `20261108`). Original actual conditional GAN generator, trained for 300 optimization steps per seed, supplied 120 generated training images per defect class for the augmentation arm. Each real-image classifier was selected by genuine validation performance; the same 360 original real test images were scored for each of **four training arms × three seeds = 12 source model tests and 4,320 actual independent-archive-persisted prediction rows**. The term *4,320 rows* counts repeated evaluations of the **same 360** real held-out photographs, and **must not be interpreted as 4,320 independent source images**.

## Actually measured results

| Original evaluated training regime | Mean real-test accuracy across 3 seeds | Mean real-test macro F1 |
|---|---:|---:|
| Real only | 0.7194 | 0.7066 |
| Horizontal flip augmentation | 0.7000 | 0.6910 |
| Conditional GAN augmented | **0.5269** | **0.5004** |
| Sample-count matched real-image repeats and flips | 0.7963 | 0.7875 |

Mean paired GAN-minus-real-only accuracy **−0.1926** (−19.26 percentage points), negative for every seed. The same-sized extra-observation comparison is a relevant control: real-image repeats + flips outperform the short-trained GAN arm by approximately **26.94 percentage points** in real held-out accuracy. **The generated images are experimental model outputs, not genuine original photographs or ground-truth physical observations.** The result provides evidence against the claimed benefit of *this short-trained GAN augmentation protocol*, **not evidence that generative augmentation never helps**.

## Newly created scientific evidence graphic

A new two-panel source-verified scientific figure `neu_original_real_seed_comparison.pdf/.png/.svg` was executed from the exact existing repository data `replicated_seed_metrics_pass13.csv`. Panel A displays all 12 original held-out accuracy values by actual training seed and method. Panel B displays the 12 original held-out macro-F1 values. The figure was visually inspected; muted four-arm color encoding, real-data axes, actual held-out sample counts, explicit negative conclusion and no generic bar charts. The *complete* reproducible figure source is committed at `code/section_8_8_2_neu_defect_benchmark/plot_original_augmentation_seed_results.py`. It does not invent uncertainty intervals, sample images, new training trials, or GAN fidelity tests. The archived original-photo prediction and defect-generation figures from Pass 13 remain referenced as original real experimental outputs.

## Proportional R1–R10 decision

- **R1 PASS:** original genuine steel-defect photography, original archive hash, duplicate detection, source split counts.
- **R2 PASS:** source-disjoint 1080/359/360 real-photo partitions; only original training portion used for GAN fitting; no test-based checkpoint selection.
- **R3 PASS:** actual conditional GAN and four real CNN classifier training runs at each of three stated seeds.
- **R4 PASS:** measured GAN-generated augmentation samples correctly labeled as generated and 4,320 saved genuine held-out test prediction rows; original model checkpoints and traces archived.
- **R5 PASS:** real-only and training-sample-count-matched real-repeat/flip quantitative controls; compare claims restricted to observed augmentation effect.
- **R6 PASS:** original macro-F1 and accuracy per three seeds, seed-level paired negative comparison and honest uncertainty limitations. No strong population significance claim from three correlated-seed evaluations.
- **R7 PASS:** original same-domain steel-surface defect images actually evaluated offline for six-class classification. No assertion of real-world deployment, physical materials properties or independent factory transfer.
- **R8 PASS:** original inference/model outputs and new varied source-number scientific figure inspected; complete source code generates the figure.
- **R9 PASS:** runnable full original GAN/CNN scripts (`run.py`, `run_multiseed.py`, analysis and validation code) and seed-level source scores committed to the proper GitHub section, original weights/checkpoints/source predictions retained in original evidence archive with provenance; final compiled chapter placement is **separately production HOLD**.
- **R10 PASS:** claim is precisely *the observed effect of GAN augmentation on held-out classification in this particular short-training-budget evaluation*. Neither generator diversity/fidelity nor deployment robustness nor other factory populations are certified.

**Key limitations:** only one NEU dataset, one fixed train/validation/test source split, just three seeds, generator trained for only 300 steps, no independent GAN sample-expert quality evaluation, some differences in total data exposures across nonmatched augmentation arms, same finite 360-image test set across seeds, and no external production-site evaluation. These limit the **scope of the scientific conclusion**; they do not invalidate the actual correctly reported result.

## Chapter certification accounting

The current `qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv` now has **four** certified narrowly scoped experiments: Level A Brownlee process emulator, Level A real handwritten-digits VAE/PCA, Level A real Mauna Loa CO₂ conditional VAE, and Level B authentic NEU steel-defect classification GAN augmentation. **No parent full Chapter 6 case-study has been silently promoted** and the previous full 14-unit chapter Master QA remains unresolved where its own titled applications require broader genuine experiments. The latest compiled manuscript itself was unavailable to inspect; its **production approval remains HOLD**.
