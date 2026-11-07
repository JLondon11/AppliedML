# Real NEU images — short pilot (2026-11-07)

Source: user-provided Kaggle archive `archive (2).zip`, SHA256 `2ccaff85d88567c52708c2f9dc4dec10cff8650749df2984f03b38716998460f`. Six classes, 1,800 source images. One exact duplicate `patches_105.jpg` = `patches_101.jpg` was excluded prior to training. Effective train/validation/test: **1080 / 359 / 360**. Original Kaggle validation set reserved as test.

Pilot only: seed 20261106, 60 conditional-GAN training steps, 3 classifier epochs, 30 generated samples/class, batch 32, CPU.

| Training arm | Test balanced accuracy | Test macro-F1 |
|---|---:|---:|
| Real-only | 0.452778 | 0.440289 |
| Real + GAN augmentation | 0.522222 | 0.490189 |

This pilot is not a converged or replicated result. No claim of general improvement, defect generation quality, or publication acceptance is supported. Longer GAN training, classifier convergence, independent baselines, repeated seeds, and generation fidelity checks are mandatory.

Execution evidence (checkpoints, real-test probabilities, generated image tensors, source manifest, figures, logs) is packaged as `Chapter6_NEU_Real_Image_Pilot_Pass12.zip` with the chapter deliverables. The Kaggle dataset itself is not committed.
