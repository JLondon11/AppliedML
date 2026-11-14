# Chapter 6 — Pass 55: authentic-photograph denoising ablation, Level A

**Decision: SCIENTIFIC_CERTIFIED for CH6-D08 only:** real-scikit-image photographic, **single-step epsilon-prediction denoising** with conditioned versus unconditioned CNNs and noisy-input/Gaussian-smoothing references. **Not** full DDPM/DDIM reverse diffusion; **not** microscopic or clinical restoration; **not** generalized model superiority. Current manuscript PRODUCTION_HOLD remains.

## Genuine data, measured protocol and retained sources

Original photographs from the scikit-image sample collection were independently loaded. Four genuinely different original images for training: `astronaut`, `coffee`, `chelsea`, `rocket`. Two separate originals withheld: `camera`, `coins`. Their SHA256 hashes after exact algorithm-prescribed grayscale conversion and resizing are held in the executed `manifest.json`; the names and original-source API are included in `code/section_6_12_2_restoration_ablation/run.py`. Artificial Gaussian corruption of *genuine source photographs* is a documented experimental intervention, not a fictional source dataset.

Independent fresh execution matched the prior archived Pass 18 numerical results: three training seeds **20261106, 20261107, 20261108**; two genuinely trained CNNs per seed (**six distinct learned network checkpoints**), **65 optimizer steps each**; held-out `camera` and `coins` at alpha-bars 0.90 and 0.75; 48 model/noise/seed/photo score rows across four arms. The two original test photos are reused across three model seeds; **48 rows are not 48 independent photographs**. Each model has an independently saved trained checkpoint and 65 actual training-loss records (390 trace records total). Fixed-budget experiments with no checkpoint selection used no validation split, and do **not** establish training convergence.

## Independently rerun original-image results

| Method | Mean PSNR (dB) | Mean SSIM |
|---|---:|---:|
| Corrupted input | 9.928145475294286 | 0.1291182247448632 |
| Gaussian smoothing, sigma 0.8 | 16.988781442011234 | **0.3177180366962218** |
| Unconditioned noise-predicting CNN | 17.352023524825412 | 0.2906048160443729 |
| Time-conditioned noise-predicting CNN | **17.68407455309745** | 0.3081990580058902 |

Across the same twelve paired original-photo/noise/seed conditions, the conditioned CNN relative to Gaussian filtering had **+0.695293 dB mean PSNR** but **−0.009519 mean SSIM**, with 9/12 PSNR improvements and 6/12 SSIM improvements. These are paired experimental conditions, not independent-population inference. The conclusion is a limited **mixed-metric** result, not a universal quality gain.

## R1–R10 proportionate requirements

| Requirement | Verdict | Basis and scope |
|---|---|---|
| R1 authentic source | PASS | Six genuine scikit-image photographs with original-source image fingerprints |
| R2 leakage/partition | PASS (method-adapted) | Four training original images versus two distinct held-out original photographs; fixed 65-step budget without validation-based checkpoint selection; no independent validation tuning or generalization claim |
| R3 executed algorithm | PASS | Six actually trained conditioned/unconditioned PyTorch CNN epsilon predictors |
| R4 saved actual outputs | PASS | 48 scored held-out real-photo conditions, 390 training trace records, six checkpoint files, real-photo denoised/gallery comparisons |
| R5 baselines/references | PASS | Corrupted original-photo input, nontrained Gaussian filter and unconditioned CNN reference |
| R6 error and uncertainty | PASS for narrow demonstration | PSNR and SSIM paired conditions, three training seeds, explicitly limited two held-out photos, mixed-metric result |
| R7 domain validation | PASS (single-step original photographic restoration only) | Actual original photographic targets and pixel/structural metrics; no microscopy or deployment assertions |
| R8 scientific figure | PASS for scoped unit | Actual original/perturbed/denoised photo gallery inspected and original-data quantitative comparison; premium non-bar plot generator committed |
| R9 code/provenance | PASS | Complete original model in `run.py`; `replay_pass55.py`, `verify_scoped_restoration.py`, `plot_measured_restoration.py` and exact numeric `pass55_measured_summary.csv` committed |
| R10 scope matching | PASS | Does not claim iterative generative diffusion, clinical validity, or general restoration superiority |

## Source code and reproducibility

Canonical tracked path: `chapters/generative_ai_part_i/code/section_6_12_2_restoration_ablation/`.
`python replay_pass55.py --output results` invokes the complete committed PyTorch model implementation, verifies real-photo hashes, per-condition records/checkpoints/figures and generates source-linked comparison figures. Requires PyTorch, NumPy, SciPy, scikit-image, Matplotlib, pandas. These are normal software/runtime dependencies, not evidence of experiment execution without checking the saved source files.

The fresh experiment output bundle retains locally produced `metrics.csv`, `training.csv`, `manifest.json`, `summary.csv`, **six checkpoint files**, and **three data-derived figures**. The executable source is in GitHub; output binary models remain in the separate delivered experiment package, not falsely claimed to be repository-stored code. Only measured summary records are additionally committed in CSV.

## Chapter accounting and release boundary

The scoped certificate ledger now contains **eight** honest scientific certificates: **seven Level A and one Level B**. This is not the number of complete 14 parent chapter case studies verified, nor proof that the current LaTeX includes the newly generated scientific figure. Final Chapter 6 scientific and production approval remain HOLD until source-to-manuscript reconciliation and missing application-specific evidence are resolved. Physical concrete test results and genuine BDD100K driving perception metrics remain absent.
