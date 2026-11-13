# Chapter 6 — Verified GitHub code inventory and scientific gates (2026-11-13)

Repository: `JLondon11/AppliedML` default branch `main`.
Verification method: GitHub repository tree and individual source/README inspection. **Code presence** does not establish **executed model evidence**.

## Verified source paths

| Study | Committed source path | Source-check conclusion | Scientific decision / limitation |
|---|---|---|---|
| Industrial original NEU steel defects | `code/section_8_8_2_neu_defect_benchmark/` | Contains `run.py`, `run_multiseed.py`, model/evidence validators, scientific figure generation and measured seed-level CSVs | Pass 50 independently certifies only an offline six-class GAN-augmentation **negative** result (Level B), not industrial deployment or generative image fidelity |
| BDD100K road-object crop augmentation | `code/section_6_driving_bdd_multiscene_validation/run.py` and `test_input_gates.py` | Runner and software-fixture tests are committed; old README's contrary claim corrected | HOLD: no genuine BDD100K trained execution or downstream held-out real-data scores reviewed; crop classification is not detection mAP |
| Generated concrete mixtures | `code/section_6_engineering_concrete_mix_generation/run.mjs`, `publication_figures.py`, `independent_lab/validate_lab_results.py` | Code, original Yeh CSV, computational summaries and blinded lab handoff exist in tree | HOLD for *physical* generated-mixture performance: Pass 42 reports **zero** cast and tested formulas; surrogate estimates are not specimen strengths |
| Industrial Brownlee process emulator | `code/section_6_28_industrial_stackloss_probabilistic_emulator/run.py` and `qa/verify_brownlee_independent_loo.py` | Source and independent verification code committed | CH6-D01 Level A scoped certificate; historical retrospective static emulator only |
| Digits VAE/PCA | `figure_generation/vae_pca_real_digits.py` and `figure_generation/vae_pca_results/` | Trained-model script and prior evidence/replay reports committed | CH6-D02 Level A scoped certificate; no general VAE superiority |
| CO₂ forecasting | `code/section_6_25_climate_co2_cvae/run.py` | Source, prediction-testing and calibration scripts committed | CH6-D03 Level A scoped certificate with explicit original 90% coverage failure |
| Digit VAE/DDPM figure baselines | `figure_generation/pass12/` | Existing VAE, DDPM, comparison and measured-metric scripts, multiple metric JSONs | Do not infer present chapter figure numbering from old `figure_20.png` or from generator filenames |

## Additional experiment code directories observed in repository tree
`section_6_12_2_restoration_ablation`, `section_6_13_1_climate_co2_cvae`, `section_6_20_1_latent_diffusion_digits`, `section_6_autonomous_scientific_discovery_nci`, `section_6_cross_domain_synthetic_evaluation`, `section_6_medical_imaging_histology_vae`, `section_6_molecular_smiles_vae`, `section_6_precision_medicine_cytology_cvae`, `section_6_real_middlebury_stereo_view`, `section_8_12_1_image_restoration`, `section_8_5_3_molecular_vae`, `section_8_8_2_texture_gan`, `section_8_8_3_gan_ablation`.

These names are **source-directory findings**, not confirmation that the corresponding *current* manuscript units, published figures and R1–R10 evidence are all mapped and scientifically certified. Some historical section 8 prefixes remain in the Chapter 6 source tree; do not rename without manuscript and provenance reconciliation.

## Key source-to-figure correction
A legacy digits script `figure_generation/pass12/real_data_figures.py` saves a robustness plot named `figure_20.png`. The earlier manuscript handoff labels Figure 6.20 an industrial-defect figure. No source inspection here establishes that the legacy digit plot is actually embedded in the current manuscript. The only defensible status is **mapping unresolved**, not proof of a manuscript figure substitution.

## Next gates
1. Reconcile all 14 full parent study titles and figure labels against latest manuscript source; preserve the four *scoped* certified experimental claims separately.
2. Obtain and execute genuine BDD100K inputs and held-out downstream evaluation before any driving-benefit claim.
3. Obtain independently signed real concrete test results before physical-performance claims; no fabricated numbers.
4. Examine exact original outputs, hashes, figures and code for each additional named unit. Run tests and scientific review before new certification.
5. Inspect and compile the latest manuscript before changing production HOLD.
