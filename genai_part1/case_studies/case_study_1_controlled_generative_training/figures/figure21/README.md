# Figure 21 — Case Study I: Training Dynamics and Convergence

Case Study figure generated from five actual VAE training runs on the public scikit-learn digits dataset.

Protocol: fixed 80/20 stratified split; seeds 21021–21025; 70 epochs; identical architecture, optimizer, batch size, and beta across runs. Shaded regions show one sample standard deviation across repeated runs. Validation uses the posterior mean to remove sampling noise from the validation trajectory.

Run `python src/figure21.py`. Outputs include PNG, SVG, per-run logs, and aggregated uncertainty data.
