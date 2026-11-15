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

## Scoped scientific certification — 2026-11-14

The **actual label-conditioned latent diffusion diagnostic on 8×8 original scanned handwritten digits** is registered as **CH6-D12, Level A SCIENTIFIC_CERTIFIED** for this limited class-conditioned generator and its original real-digit-trained classifier evaluation. Original-source sklearn digits (1,797 genuine scans; 1257/270/270 train/validation/test), trained VAE and two diffusion variants, 1,000 actual generated outputs per arm, 2,000 individual saved probability rows, original model checkpoints and authentic model-output figures were verified in earlier Pass23/47 reports. The classifier agrees with requested label on **778/1000** conditional samples and **94/1000** unconditional samples. These are not independent externally labeled real-image generalization measurements; this is an indirect diagnostic of actual generated digits and one model-training seed. Source class summary and its per-class totals are mechanically reconciled by `qa/validate_scoped_ch6_certifications.py`.

**Do not state that this experiment is Stable Diffusion, text-prompt generation, classifier-free guidance, natural-image text-to-image synthesis, or full held-out generative-fidelity certification.** This certificate does not promote that broader chapter parent application, and final compiled manuscript figure placement is separately **PRODUCTION_HOLD**.
