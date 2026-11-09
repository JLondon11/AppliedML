# Chapter 9 — Matched DDPM versus DDIM using real handwritten digits

This evaluation reuses the **same 45-epoch DDPM noise-prediction checkpoint**, trained on 1,257 real handwritten-digit images from `sklearn.datasets.load_digits`. Its 270-image validation set selected the checkpoint, while a separate 270-image test set serves as an independent real-data reference. No DDIM network was trained independently: DDIM changes only the reverse sampler.

The script runs ancestral DDPM with 100 steps and deterministic DDIM (`eta=0`) with 100, 50 and 20 steps. Each uses the same initial Gaussian noise for a given random seed. Three seeds (201, 202, 203), each generating 128 images per method, support a simple repeated-seed uncertainty report.

## Experimental results

| Method | Mean nearest-training Euclidean distance (± SD across seeds) | Fraction of pixel outputs outside [-1,1] |
|---|---:|---:|
| DDPM-100 | 4.204 ± 0.142 | 0.293 |
| DDIM-100 | 7.012 ± 0.253 | 0.450 |
| DDIM-50 | 7.041 ± 0.254 | 0.453 |
| DDIM-20 | 7.138 ± 0.256 | 0.466 |

**Interpretation:** These metrics describe raw outputs in normalized pixel space. A smaller distance to a training image does not prove greater visual quality or generalization; it may also reflect poor diversity or memorization. Display clipping is applied only when rendering images, never during metric computation. This particular small model and noise schedule exhibit substantial quality limitations, particularly for DDIM. Do not generalize these rankings to other trained diffusion models.

## Reproduce after training the DDPM

```bash
python chapters/generative_ai_part_i/figure_generation/pass12/ddpm_digits_experiment.py --out chapters/generative_ai_part_i/figure_generation/pass12/outputs/ddpm --epochs 45
python chapters/generative_ai_part_i/figure_generation/pass12/ddim_ddpm_comparison.py --checkpoint chapters/generative_ai_part_i/figure_generation/pass12/outputs/ddpm/best_ddpm.pt --out chapters/generative_ai_part_i/figure_generation/pass12/outputs/ddim_comparison
python -m unittest discover -s tools/scientific_qa -p 'test_*.py'
```

Outputs include `comparison_metrics.csv`, `comparison_metrics.json`, `ddpm_ddim_real_digits.png`, and `ddpm_ddim_diagnostics.png`. GitHub Actions uploads generated assets; binary model checkpoints and generated plots should not be represented as having been committed to GitHub unless separately verified.

## Suggested text for the book

A deterministic DDIM reverse step can be written as

```text
x0_hat = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta(x_t,t)) / sqrt(alpha_bar_t)
x_prev = sqrt(alpha_bar_prev) * x0_hat + sqrt(1 - alpha_bar_prev) * epsilon_theta(x_t,t)
```

with `alpha_bar_prev=1` at the final step and `eta=0`. This permits subsampling the original noise schedule while using the same trained epsilon predictor. In the real-digits comparison, DDIM was evaluated at 100, 50 and 20 steps against a 100-step DDPM sampler. The observed raw-pixel diagnostics favored DDPM in this limited run, and all settings produced imperfect images. This study demonstrates mathematical sampling differences and measured limitations, not a state-of-the-art performance comparison.
