# Figure 20 — Domain-Constrained Generative Modeling for Scientific or Medical Data

Application figure generated from executable code using the public Wisconsin Diagnostic Breast Cancer dataset distributed by scikit-learn. A VAE is trained only on the fixed training split (seed 20020). The figure evaluates generated tabular observations using standardized feature means, variance ratios, feature-correlation preservation, and PCA distribution geometry.

This is a reproducible methodological experiment; it is not a clinical validation study and makes no diagnostic-performance claim.

Run `python src/figure20.py`. The script writes PNG, SVG, feature metrics, summary metrics, and training logs to `outputs/`.
