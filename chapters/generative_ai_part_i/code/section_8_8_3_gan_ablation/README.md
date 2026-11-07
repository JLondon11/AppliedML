# Section 8.8.3 — Controlled DCGAN horizontal-flip ablation

This executed, controlled study evaluates the effect of horizontal-flip augmentation on 32×32 grayscale DCGAN texture synthesis from genuine `skimage.data` brick, gravel, and grass photographs. **None of these images is labelled as an industrial defect.** The experiment does not establish defect generation or downstream inspection performance.

## Execute
```bash
python chapters/generative_ai_part_i/code/section_8_8_3_gan_ablation/run.py --steps 120
python chapters/generative_ai_part_i/code/section_8_8_3_gan_ablation/test_outputs.py
```
Dependencies: torch, scikit-image, numpy, matplotlib. `texture_gan_reference.py` is the pinned architecture/metric implementation. All output files are written into `results/`. Binary model checkpoints, generated patches, figures and full training logs are available in the Pass 9 execution ZIP, and are not committed in this source-only update.

Data: one real 512×512 image per texture. Left 384 pixel columns supply 713 overlapping training patches; right 128 columns provide 217 spatially held-out patches. Both ablation arms use 120 steps and the same generator/discriminator initialization seed, but the augmentation arm consumes additional random draws, so the subsequent stochastic training trajectory is not perfectly paired. Evaluation latent sampling uses a fixed seed. One seed, no confidence interval.

## Actual observed results
Metric: Euclidean distance between standardized *mean* texture features of generated patches and held-out photographic patches (lower better). This is not FID.

| Texture | No augmentation | Horizontal flips (p=0.5) |
|---|---:|---:|
| Brick | 11.681413 | 10.493758 |
| Gravel | 26.291688 | 17.293800 |
| Grass | 19.412618 | 16.421878 |

All three scores improve under augmentation at the chosen 120-step budget, but remain weak in absolute magnitude. Training only 120 steps causes underfitting and limited image realism; do not infer general improvements without repeated seeds and a substantive defect benchmark. The archived section 8.8.2 experiment ran 240 steps and is not directly comparable as an augmentation effect.

## Required for complete defect case study
Run a GAN and downstream classifier/segmenter on an approved, annotated industrial dataset (for example NEU-DET), use disjoint train/validation/test evaluation, report class-specific metrics and GAN sample quality, and preserve the actual checkpoint, dataset manifest and numerical results. This remains pending.
