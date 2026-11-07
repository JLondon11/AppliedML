# Section 8.8.2 — Real photographic texture GAN diagnostic

**Scope:** An executed DCGAN proof-of-concept using real grayscale images of brick, gravel and grass from `skimage.data`. These are natural/material textures, not annotated industrial defects. The result must not be described as successful defect generation, defect segmentation or anomaly detection.

Execute: `python chapters/generative_ai_part_i/code/section_8_8_2_texture_gan/run.py --steps 240`.

Three independent 32×32 patch GANs train on the left three-quarters of each 512×512 source photograph; the right quarter provides a spatially separate evaluation area. 713 overlapping training patches and 217 overlapping held-out patches are extracted per photo. The code produces generated images, per-step GAN loss curves, a texture feature comparison figure, generated arrays, model checkpoints, metrics.csv, and a source-image SHA-256 manifest.

**Measured standardized mean-texture-feature distance, lower better** (not FID):

| Texture | GAN | Constant-intensity baseline | Real training patches reference |
|---|---:|---:|---:|
| Brick | 2.356375 | 7.246906 | 0.695720 |
| Gravel | 5.344645 | 18.291330 | 0.198051 |
| Grass | 7.123228 | 13.560668 | 0.992984 |

The GAN is better than a deliberately weak constant-mean baseline, but is substantially worse than real training patches. This metric describes only aggregate pixel mean, standard deviation, gradient magnitudes and nearest-pixel correlation; it does not establish visual diversity, semantic texture fidelity or anomaly usefulness.

**Validation limitations:** One photograph per texture, overlapping spatially correlated patches, no train/validation model selection, no defect annotations, no FID/KID, and no independent replicated experiments. The full industrial-defect case study remains **HOLD** until a real annotated defect dataset and a relevant downstream model benchmark are executed.

Local execution output is supplied separately in the book evidence ZIP; binary checkpoints and image outputs are not committed in this code-only GitHub pass.
