# NEU-DET real steel-surface defect classification and GAN augmentation experiment

## Scientific status and claims

**Executed original real-data benchmark:** Three-seed genuine NEU-DET classification experiment completed and documented in `REPLICATED_BENCHMARK_PASS13.md` (2026-11-07). The earlier Pass 10 planning/smoke-test status has been **superseded** by these real-data measurements. The original Pass 10 fixture was software-only and MUST NOT be cited as empirical NEU data.

**Scoped scientific verdict:** `CH6-D04`, **Level B SCIENTIFIC_CERTIFIED** solely for an *offline six-class real-steel-photograph classification experiment comparing four augmentation training arms under a fixed short conditional-GAN training budget*. This does not certify generator fidelity for new defect photographs, downstream production deployment, anomaly localization, cross-factory robustness or defect-free-versus-defect binary screening. It is also not evidence that GAN augmentation universally fails. **Final manuscript production status: HOLD**; the exact current industrial-defect Figure 6.20 asset/page mapping is unverified.

## Original source and leakage control

Original NEU surface-defect dataset: 1,800 grayscale 200×200 photographed defects, six original classes (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches), 300 nominal per class. User-supplied original dataset archive SHA256:
`2ccaff85d88567c52708c2f9dc4dec10cff8650749df2984f03b38716998460f`.
One duplicate image (`patches_105.jpg`, matching `patches_101.jpg`) was excluded from the development partition. Original source split **1,080 training / 359 validation / 360 distinct held-out real test files**. Models and any generated training supplements are fitted on original training data only. The same real test photographs are evaluated across three seeds; **4,320 prediction rows across twelve model configurations does not equal 4,320 independent test images**.

## Actual executed experiment

Three seeds: `20261106`, `20261107`, `20261108`. Same architecture and 12 classifier-training epochs, selected by original validation balanced accuracy. Conditional DCGAN: 300 optimizer steps per seed; 120 generated training images per each of six source classes. Compared these actual training arms:

- Real original photographs only
- Real original photographs with horizontal-flip augmentation
- Real originals plus short-trained GAN-generated images
- Real originals plus *sample-count-matched* resampled original training photographs and flips

| Actually measured arm (360 original held-out photographs) | Test accuracy mean (3 seeds) | Macro-F1 mean (3 seeds) |
|---|---:|---:|
| Real only | **0.7194** | **0.7066** |
| Horizontal flips | **0.7000** | **0.6910** |
| GAN augmentation | **0.5269** | **0.5004** |
| Matched-count real-photo repetition + flips | **0.7963** | **0.7875** |

The GAN-minus-real-only paired seed difference in held-out accuracy is **−0.1926** (−19.26 percentage points) in this limited protocol, negative for all three executed seeds. The matched-size real-image control outperformed the GAN arm by approximately 26.94 percentage points. The same finite test data across seeds and the 300-step GAN budget limit external or population interpretations. GAN outputs were generated model images, **not newly observed photographed defects**. The negative result is scientifically valid and must not be presented as GAN augmentation improving inspection.

## Full executable source and original evidence

- `run.py`: model classes, genuine data loading and original source partition checks.
- `run_multiseed.py`: actual conditional-GAN fitting and three-seed held-out classification evaluation, original generated-image galleries, checkpoints, per-photo probabilities and training traces.
- `run_repeat_control.py`: the genuine sample-count-matched original image repetition control.
- `analyze_multiseed.py` and `verify_evidence.py`: analysis and output audit.
- `replicated_seed_metrics_pass13.csv` and `replicated_metrics_pass13.csv`: recorded genuine source-derived measured results, **not simulation**.
- `plot_original_augmentation_seed_results.py`, `publication_figures.py`: source-driven scientific result plots. Sample photographic/generated output galleries are part of the original Pass 13 evidence archive; they must be inspected before any new exact manuscript figure identification.
- `REPLICATED_BENCHMARK_PASS13.md`: initial full original experiment and scientific limitations.

Regeneration from authentic real data, with the expected original source-directory layout:

```bash
python chapters/generative_ai_part_i/code/section_8_8_2_neu_defect_benchmark/run_multiseed.py \
  --data-root /path/to/authentic/NEU-DET
```

**Archived original execution and binary completeness:** the original Pass 13 deliverable contained the true benchmark predictions, GAN/classifier checkpoints, original source-manifest and actual prediction photos. Those binaries are not all available in the default-branch tree. The persisted aggregate metrics alone are insufficient for independent confusion-matrix recomputation; future true GitHub Actions release artifacts must preserve test-row predictions and image-output panels. Do not infer original figures from a similarly numbered handwritten-digit file.

## Honest Figure 6.20 production gate

The user-supplied *Pass 37 handoff* states that the manuscript's Figure 6.20 is an **industrial-defect figure placed within the corresponding case study, on printed page 67** of the previously built 109-page PDF. It confirms historical placement only. The current full Chapter 6 manuscript binary is not accessible here; a provisional script called `figure_20.png` under a different handwritten-digit study is **not** proof of manuscript identity. Do not replace Figure 6.20 with that digits figure or claim this README proves its exact caption/source or that its latest typesetting has passed.

**Full parent-application and printed-production QA remain HOLD** wherever broader stated application claims require actual independent original results or full compiled manuscript inspection.

## Matched-seed uncertainty and negative-result integrity — Pass 67

Source: \`replicated_seed_metrics_pass13.csv\`, the **twelve previously executed genuine NEU held-out classifier results** (three training seeds × four training arms), original source SHA-256 \`1f74ab0c8067612a072c2eff5554b971682ce851bd4c7ca5eb585fbb5f163d9e\`. All four arms are evaluated on the **same 360 actual held-out original photographs** in every seed; these are neither newly collected images nor independent across seed repetitions.

The new complete \`audit_paired_seed_uncertainty.py\` checks that every original seed/arm is present exactly once, reads the real numeric accuracy and macro-F1, and computes **paired within-seed** GAN-minus-control effects. The honest result includes:

| Original measured paired effect (GAN minus control) | Across-seed mean | Individual observed seed effects |
|---|---:|---|
| Accuracy vs real-only | −19.2593 percentage points | −18.6111, −19.4444, −19.7222 pp |
| Macro-F1 vs real-only | −20.6174 pp | −20.6921, −22.1786, −18.9814 pp |
| Accuracy vs matched real-repeat/flip | −26.9444 pp | −24.7222, −22.5000, −33.6111 pp |
| Macro-F1 vs matched real-repeat/flip | −28.7086 pp | −27.1225, −24.7150, −34.2882 pp |

All **three** recorded GAN-minus-control differences are negative for each listed comparison. The independent local scientific calculation reproduced the original source means; an exact **one-sided three-pair sign test yields p=0.125**, not p<0.05. A conventional Student-t 95% interval can be calculated with 2 degrees of freedom and is **descriptive and fragile** with only three seeds. The seed-level results do **not** establish independent replication across factories, model budgets or train/test datasets, and this source CSV lacks per-image paired predictions for McNemar inference. Do not treat 4 arms × 3 seeds × 360 predictions as 4,320 independent source photographs.

New reproducible code and results are organized with the actual experiment:

- \`audit_paired_seed_uncertainty.py\` — standard-library paired comparisons, small-sample sign inference, transparent descriptive t intervals and rigorous observed-source schema checks.
- \`paired_seed_science/paired_seed_statistics.json\` and \`paired_seed_science/paired_seed_effects.csv\` — source-derived numerical evidence, preserved by seed and metric.
- \`paired_seed_science/paired_seed_effects.svg\` — restrained vector paired-result figure with **actual measured per-seed effect dots**, not a generic bar chart or invented observations.
- \`plot_neu_paired_seed_evidence.py\` — alternative publication-readable source-linked single-panel real-difference figure code.
- \`test_neu_paired_seed_uncertainty.py\` — seven tests including original source integrity, invalid probabilities, missing/duplicate records and incorrect source seeds.
- The scoped Chapter 6 science validator now independently invokes this measured-source paired effect auditor. CI is configured to rerun all the genuine-data calculations and software tests, with result integrity checks.

**Scientific certification remains CH6-D04 Level B within the original real NEU augmentation experiment, documenting its negative result.** The full industrial-defect case-study manuscript, its actual Figure 6.20 publication asset/caption, cross-factory domain validation and final Chapter 6 production pass remain separately **HOLD**.

### Native print-width scientific figure

The wide `paired_seed_science/paired_seed_effects.svg` is a truthful measured-data scientific source illustration but is **not** certified for normal-width print: its small type scales poorly to 6.5 inches. Use the separately executed `plot_neu_paired_seed_print.py` to regenerate `paired_seed_science/neu_paired_seed_print_468pt.svg`, an independently source-reconciled **468pt-wide** figure with **10.5pt minimum vector text** and no visible clipping in its reviewed native-width rendering. Its actual source-generated Git blob SHA-1 is `05277a839971501bed5665186b69f4d287b88e2f` and the CI workflow checks exact SVG regeneration and type size.

**Editorial boundary:** This is a new original-results figure candidate, **not** a replacement automatically assigned to Figure 6.20. It has no verified manuscript figure number, caption placement or printed page in the absent current Chapter 6 PDF/TeX. Staging ledger: `quality_assurance/pass59_image_first_figure_staging.json`. A true final production approval requires the complete actual manuscript.
