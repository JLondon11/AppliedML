# Section 8.12.1 — Diffusion noise-prediction diagnostic

This directory implements a **time-conditioned epsilon-prediction CNN** for *single-step denoising* on genuine example photographs from scikit-image. It is not a complete DDPM sampler and does not measure generative sample quality.

## Execution

Install Python 3 with numpy, scipy, scikit-image, matplotlib and torch; then run from any directory:

```bash
python chapters/generative_ai_part_i/code/section_8_12_1_image_restoration/run.py
```

Outputs are written to `results/`: `metrics.csv`, `training_loss.csv`, `manifest.json`, `checkpoint.pt`, and two `figures/*.png` files. Preserve these outputs with the chapter evidence package; the large checkpoint and figures are not committed in this source-only commit.

## Data and evaluation

Training photographs: astronaut, coffee, chelsea, rocket. Held-out test photographs: camera, coins. All are from `skimage.data`, resized to 128 × 128; random 32 × 32 crops train the model. Fixed seed 20261106, 420 training steps, Adam, batch 32. Fixed alpha-bar levels 0.90 and 0.75 define corruption. Report PSNR and SSIM against reference images. Comparator: a preselected Gaussian filter of sigma 0.8 applied after rescaling the noisy input.

No validation split or hyperparameter search is claimed. There are only two held-out photographs. Measurements demonstrate behavior of this *diagnostic* and must not be used as evidence of DDPM/score-based generative superiority.

The accompanying `metrics_reference.csv` is the output of the independently executed Pass 6 script with the same algorithm and seed; run the committed script to regenerate and compare. Runtime library versions can affect exact floating-point results.

## Acceptance boundary

This implementation documents only section 8.12.1. It does **not** certify the chapter's other 13 case studies/applications. Those require independently executed, domain-specific data pipelines and measured results.
