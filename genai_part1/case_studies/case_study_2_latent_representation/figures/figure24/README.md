# Figure 24 — Case Study II: Latent-Space Organization

Case Study figure generated from five actual VAE training runs on the public scikit-learn digits dataset.

Protocol: fixed stratified 80/20 split; seeds 24024–24028; 85 epochs; 12-dimensional latent representation. Panel (a) uses PCA only to display the held-out 12-D posterior means. Panel (b) reports held-out 15-nearest-neighbor class purity using neighbors fitted only on training-set latent means. Quantitative silhouette and between/within separation metrics are computed in the full 12-D latent space, not in the 2-D display projection.

Run `python src/figure24.py`. Outputs include PNG, SVG, per-seed metrics, and aggregate uncertainty results.
