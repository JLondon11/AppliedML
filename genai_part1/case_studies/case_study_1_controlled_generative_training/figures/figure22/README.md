# Figure 22 — Case Study I: Quantitative Model Comparison

Case Study figure from actual matched-protocol experiments on the public scikit-learn digits dataset.

Models: latent autoencoder baseline, VAE, and GAN. Each is trained for 45 epochs under three seeds (22022–22024), on the same fixed 80/20 split and preprocessing. Support precision and recall use a nearest-neighbor threshold determined only from the real training distribution. Parameter counts, CPU training time, and sampling time are recorded by the script.

The zero support scores observed for the compact GAN are retained as measured rather than cosmetically altered; they indicate failure under this small CPU-scale protocol and are not a general claim about GAN performance.

Run `python src/figure22.py`. Outputs include PNG, SVG, per-run measurements, and aggregate uncertainty results.
