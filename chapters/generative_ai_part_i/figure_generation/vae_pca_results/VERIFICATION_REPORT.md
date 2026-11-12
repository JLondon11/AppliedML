# Chapter 6 VAE/PCA evidence verification — 2026-11-12

## Scope and verdict

**Eleven independent consistency checks passed (0 failed)** on the locally saved real-digit VAE/PCA result bundle. This is a source-artifact and numerical-output consistency check, **not** a model retraining or complete case-study certification. The current Pass-37 manuscript has not been inspected; no claim is made that this figure has been included in it.

## Actual recorded data and evaluation

- Source: scikit-learn handwritten digits (1,797 original 8x8 digit observations).
- Split: train 1,257, validation 270, independent test 270; test contains 270 unique source indices.
- Test-index SHA-256: `bfe1805d067fe13271534f968cdf891b227872c039fbc2151f7772fedd394b93`.
- Training history: 35 epochs, checkpoint chosen using validation reconstruction MSE.
- VAE mean test reconstruction MSE: **0.0538716204** (recorded).
- Two-component PCA mean test reconstruction MSE: **0.0531239174** (recorded).
- Mean paired PCA minus VAE error: **-0.0007477068**.
- Paired 3,000-draw bootstrap 95% interval: **[-0.0015158851, 0.0000040249]**.
- Interpretation: this run does not establish a VAE improvement over PCA; interval includes zero.

## Recomputed audit checks

1. Dataset counts, 270 test rows and 35 recorded epochs.
2. Unique, in-range original image indices.
3. Recorded test-index hash.
4. Finite nonnegative per-image reconstruction MSE.
5. Rowwise PCA minus VAE differences.
6. Recomputed VAE test mean.
7. Recomputed PCA test mean.
8. Recomputed paired mean.
9. Recomputed bootstrap interval using the recorded seed and sampling procedure.
10. Valid PNG image with dimensions 2400x1280.
11. A nonempty trained VAE checkpoint.

## Evidence artifact SHA-256 checksums

| File | SHA-256 |
|---|---|
| real_digits_vae_pca_evaluation.png | `3a70880b9822996d114bf34c5829bf7dabfaa08fb548df4e91c15524e229f4ee` |
| vae_checkpoint.pt | `b5df83ea5f95600b836af38295a93e9424630edafd2938a694ba3db2f5bd5431` |
| training_history.csv | `f0b1f478e4b0afc1b8927263190f4669b334f9bf869388717e4a90ddb0112599` |
| test_predictions.csv | `aac7673654649f097f87d4f2e17e7d15ca63e60223b67504810db970fcd026e2` |
| metrics.json | `232580db585b8e68c244e640c716e13480b40e8fdf8e75007ec19a5b04c76e07` |

## Reproducibility

- Executed experiment source: `../vae_pca_real_digits.py`.
- Automated audit source: `../audit_vae_pca_evidence.py`.
- Locally saved result bundle: `Chapter6_RealDigits_VAE_PCA_Evidence.zip`.
- Audit run command: `python audit_vae_pca_evidence.py --results <unzipped>/outputs/vae_pca --report audit.json`.

## Remaining limitations

- The bootstrap reflects variation among test observations conditional on a single trained model, not seed-to-seed training uncertainty.
- Printed figure numbering and caption claims in the current chapter are unverified.
- The figure in the original evidence bundle predates a subsequent script-only palette revision in GitHub. Rebuild/review the figure from the final committed script before citing identical-image provenance.
- The complete artifact bundle (checkpoint, predictions, figure) has **not** been verified as committed to GitHub. Its SHA inventory here is not equivalent to a GitHub file or release.
- Do not infer validity of the industrial-defect, driving or concrete-mixture experiments from this separate handwritten-digit demonstration.
