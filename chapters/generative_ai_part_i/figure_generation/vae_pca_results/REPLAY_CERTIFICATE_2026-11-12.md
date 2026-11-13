# Archived VAE–PCA experiment — independent exact replay

## Execution and outcome

An independent replay of the supplied archived experiment was completed with the *archive's source* `vae_pca_real_digits.py`, seed 19, 35 epochs, and output path isolated from the original evidence bundle. The program ran successfully, and **every primary output file was byte-for-byte identical to the archived reference** (SHA-256 equal).

The input archive was `Chapter6_RealDigits_VAE_PCA_Audited.zip`. The archived Python source SHA-256 was `236779da4efce2ce2be16eb9be819ca93071d5b8889b875083930cd4a59737ba`.

| Compared artifact | Original SHA-256 | Replay |
|---|---|---|
| `metrics.json` | `232580db585b8e68c244e640c716e13480b40e8fdf8e75007ec19a5b04c76e07` | exact |
| `test_predictions.csv` | `aac7673654649f097f87d4f2e17e7d15ca63e60223b67504810db970fcd026e2` | exact |
| `training_history.csv` | `f0b1f478e4b0afc1b8927263190f4669b334f9bf869388717e4a90ddb0112599` | exact |
| `vae_checkpoint.pt` | `b5df83ea5f95600b836af38295a93e9424630edafd2938a694ba3db2f5bd5431` | exact |
| `real_digits_vae_pca_evaluation.png` | `3a70880b9822996d114bf34c5829bf7dabfaa08fb548df4e91c15524e229f4ee` | exact |

The archived metrics describe 1,797 real handwritten-digit samples, with 1,257 training / 270 validation / 270 held-out test samples. The VAE test MSE is 0.0538716204 and the PCA reference MSE is 0.0531239174. The paired mean difference (PCA minus VAE) is -0.0007477068, bootstrap 95% interval [-0.0015158851, 0.0000040249]. **This is not evidence of VAE superiority.**

## Important provenance qualification

The archived source already contains a muted categorical palette. The replay validates the **archived source and archived figure**; this audit did not perform a bytewise comparison of the archive's source with the current GitHub head revision of `figure_generation/vae_pca_real_digits.py`. Consequently, head-revision identity and the current head's output hash remain unverified until a source comparison and head-revision replay are completed.

## Certification boundary

This is an **exact execution replay for one focused Level A experiment**. It does not verify the Pass-37 manuscript, printed figure numbering, current figure captions, or any industrial-defect, synthetic-driving or concrete-mixture results. Full unit certification remains pending figure-manuscript reconciliation and artifact-delivery checks.

## Re-run command

```bash
python vae_pca_real_digits.py --out replay --seed 19 --epochs 35
sha256sum replay/metrics.json replay/test_predictions.csv replay/training_history.csv replay/vae_checkpoint.pt replay/real_digits_vae_pca_evaluation.png
```
