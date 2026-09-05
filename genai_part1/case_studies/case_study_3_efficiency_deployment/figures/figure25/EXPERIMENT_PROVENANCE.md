# Figure 25 CPU experiment provenance

This case-study figure was generated from an actual CPU experiment executed in the ChatGPT Python runtime.

- Dataset: public sklearn digits dataset (real observations)
- Generative model: compact denoising diffusion model trained during the experiment
- Samplers: DDPM and DDIM
- Step budgets: 10, 20, 40, 80
- Repeats: 3
- Generated observations per operating point per repeat: 1,200
- Quality metric: Fréchet distance in a separately trained 48-dimensional classifier feature space; lower is better
- Classifier held-out test accuracy: 0.9644
- Efficiency: measured CPU wall-clock sampling latency and exact denoiser evaluations per sample
- Dominance: Pareto dominance computed jointly over measured latency and feature Fréchet distance
- Figure artwork: native Matplotlib SVG plus PNG preview; no embedded figure title/caption and no invented measurements

The CSV files contain the measured values used by the figure.
