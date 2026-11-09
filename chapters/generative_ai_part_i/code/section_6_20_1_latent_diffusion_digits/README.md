# Chapter 6 §6.20.1 — Class-conditioned latent diffusion on real digit images

**Scope:** Executed VAE latent diffusion model with a class-label embedding (0–9). **Not** pretrained Stable Diffusion, natural-language text-to-image, cross-attention, or classifier-free guidance.

Dataset: genuine handwritten digit images distributed by `sklearn.datasets.load_digits()`; 1,797 images of size 8×8, stratified split 1,257 training / 270 validation / 270 untouched test. Fixed seed 20261109. Sixteen-dimensional VAE, 25 epochs (validation checkpoint), 48-step beta schedule, 35 epochs per diffusion variant, and 1,000 sampled images per variant. Independent logistic evaluator trains **only** on original training images, achieving 95.19% accuracy on untouched genuine digit test images.

| Evaluation | Conditional | Unconditional |
|---|---:|---:|
| Generated sample matches requested digit by real-trained classifier | 77.8% | 9.4% |
| Mean closest real training-image pixel MSE | 0.01980 | 0.02337 |

Held-out VAE reconstruction RMSE: 0.162. All images, classifier probabilities, two diffusion checkpoints, VAE checkpoint, split indices, model training histories, manifest, figures, and verified source script are in the Pass 23 chapter ZIP.

Install `torch`, `numpy`, `scikit-learn`, `matplotlib`; run `python run.py --vae-epochs 25 --diffusion-epochs 35` then `python test_outputs.py`. The generated NumPy-array checkpoint artifacts must only be opened from trusted local runs; the verification script uses `weights_only=False` for those generated NumPy objects.

**Limitations:** one model seed; low-resolution monochrome digits; label embeddings rather than natural language; no trained pixel-space diffusion control; classifier evaluation is only an indirect measure of semantic quality. The manuscript Stable Diffusion case study has **partial real-data evidence** and remains **HOLD** for complete scientific certification.

Chapter manuscript insertion: Figure 6.43, Figure 6.44 and Table 6.25 in the Pass 23 edition. Full data and generated results are in the downloadable chapter ZIP, not committed as binary tensors to GitHub.
