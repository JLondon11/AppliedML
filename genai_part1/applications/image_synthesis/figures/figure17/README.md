# Figure 17 — Image Synthesis Across Generative Model Families

**Section:** Applications — Image Synthesis  
**Classification:** Application figure

This is an actual executable experiment, not an AI-generated illustration. It uses the public `sklearn.datasets.load_digits` dataset and trains three small generative models from scratch: a VAE, a GAN, and a DDPM-style noise-prediction diffusion model.

The rendered rows contain authentic dataset observations and actual outputs sampled from the trained models.

## Reproduce

```bash
python -m pip install -r ../../../../requirements.txt
python src/figure17_image_synthesis.py
```

Seed: `17017`. The script saves the publication PNG/PDF, exact sampled tensors, and training logs under `outputs/`.

The models are intentionally compact CPU-scale implementations for transparent reproducibility. The figure must not be represented as a state-of-the-art benchmark comparison.
