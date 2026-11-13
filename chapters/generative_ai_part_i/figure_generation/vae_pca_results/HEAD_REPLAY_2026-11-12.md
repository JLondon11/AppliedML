# Chapter 6 — Current plotting configuration replay and hash verification

## Result
The local archived experiment was reconstructed with the **same muted ListedColormap ten-color configuration found in the current GitHub source**. The resulting Python implementation was executed using seed 19 and 35 epochs, and the five resulting evidence files each had SHA-256 hashes identical to the retained replay. This validates the current plotting configuration and recorded calculation in the tested environment. The literal current GitHub source file was not independently downloaded into the execution environment; therefore this is a **configuration-equivalent head replay**, not a byte-for-byte source-checkout attestation.

## Evidence comparison
| Artifact | Archived SHA-256 | Configuration-equivalent replay |
|---|---|---|
| `metrics.json` | `232580db585b8e68c244e640c716e13480b40e8fdf8e75007ec19a5b04c76e07` | exact |
| `test_predictions.csv` | `aac7673654649f097f87d4f2e17e7d15ca63e60223b67504810db970fcd026e2` | exact |
| `training_history.csv` | `f0b1f478e4b0afc1b8927263190f4669b334f9bf869388717e4a90ddb0112599` | exact |
| `vae_checkpoint.pt` | `b5df83ea5f95600b836af38295a93e9424630edafd2938a694ba3db2f5bd5431` | exact |
| `real_digits_vae_pca_evaluation.png` | `3a70880b9822996d114bf34c5829bf7dabfaa08fb548df4e91c15524e229f4ee` | exact |

## Limits
- The current print manuscript and all Figure 6.20 asset/caption hashes remain unavailable. Do not treat this VAE/PCA validation as evidence about industrial-defect imagery.
- This is one run with one random seed on the real handwritten-digits dataset, with 1,257 training / 270 validation / 270 independent-test examples. Metrics were VAE MSE 0.0538716204 versus PCA MSE 0.0531239174; no superiority claim is established.
- Binary evidence assets must be present in GitHub, as suitably managed artifacts, or preserved in a versioned release before code-delivery gate R9 can be closed for the complete unit. This report only certifies hashes observed from local files.
- R8 manuscript integration and section title/claims remain gated until the current chapter source/PDF and its exact figure files are inspected.

## Required release checks
1. Inspect current manuscript figure-to-generator mappings and captions.
2. Obtain archived and current figure assets and compare SHA-256; prevent unrelated generator outputs from being assigned to Figure 6.20.
3. Review each application-specific dataset, execution logs, and downstream evaluations independently.
