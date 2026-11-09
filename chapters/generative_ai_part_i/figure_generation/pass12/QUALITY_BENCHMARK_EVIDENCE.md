# Chapter 9 — Real-data DDPM/DDIM generative-quality audit

This experiment uses the trained DDPM epsilon-prediction checkpoint from the real `sklearn.datasets.load_digits` handwritten-digit dataset (1,797 actual 8×8 digit images), with disjoint train/validation/test partitions of **1,257/270/270**. The same DDPM-trained model is sampled using ancestral DDPM (100 steps) and deterministic DDIM (100, 50, and 20 steps). Three seeds (201, 202, 203) produce 128 generated images per method per seed.

A separate 32-hidden-unit MLP digit classifier is fitted **only on the real training split** and achieves 96.30% accuracy on the held-out real test images. Its learned features are used for an explicitly domain-specific Fréchet diagnostic; this must **not** be called canonical FID or Inception Score.

| Distribution | Digit-classifier feature Fréchet | Mean classifier confidence |
|---|---:|---:|
| Real vs. real held-out reference (128/128) | 7.671 | — |
| DDPM 100-step | 54.363 | 0.9173 |
| DDIM 100-step | 99.802 | 0.9308 |
| DDIM 50-step | 98.702 | 0.9318 |
| DDIM 20-step | 95.146 | 0.9327 |

Values for generators are means across three seeds. Full per-seed values and uncertainty estimates are written to `quality_per_seed.csv` and `quality_metrics.json`. The real/real reference uses two disjoint held-out sets of 128 images to provide a same-sample-size calibration.

### Scientific interpretation

Even though generated samples often receive high classifier confidence, their feature distributions differ substantially from held-out real digits. The original raw generated pixel values also exhibit substantial out-of-range fractions (approximately 29.3% for DDPM and 45–46.6% for DDIM) **before** display clipping. Clipping is applied only for classifier-feature evaluation; raw-range violations are retained as separate diagnostics. Classifier confidence is not visual fidelity; feature-space proximity is not proof of memorization; and a three-seed, 8×8-image experiment is not a comprehensive general quality benchmark.

The quality audit is an **additional figure candidate**, not yet proven to be inserted into the chapter manuscript. All numerical results were obtained from actual real-data-trained checkpoint inference.

### Reproduce

```bash
python chapters/generative_ai_part_i/figure_generation/pass12/quality_benchmark.py \
  --checkpoint chapters/generative_ai_part_i/figure_generation/pass12/outputs/ddpm/best_ddpm.pt \
  --out chapters/generative_ai_part_i/figure_generation/pass12/outputs/quality
python -m unittest discover -s tools/scientific_qa -p 'test_*.py'
```

The workflow publishes `quality_metrics.json`, `quality_per_seed.csv`, and `quality_diagnostics.png` as output artifacts. A passing remote workflow and full manuscript integration remain to be independently confirmed.
