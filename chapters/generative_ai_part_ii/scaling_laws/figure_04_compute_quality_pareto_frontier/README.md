# Figure 4 — Compute–Quality Pareto Frontier

Chapter: Generative AI Part II  
Section: Scaling Laws

Reproduction based on the Approach 3 parametric scaling-law fit reported by Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (arXiv:2203.15556).

Published fit used:

L(N,D) = 1.69 + 406.4 N^(-0.34) + 410.7 D^(-0.28)

with training-compute approximation C ≈ 6ND.

The solid curve is the analytically derived compute-optimal allocation. Secondary curves are controlled counterfactual allocations evaluated with the same published fit; they are not measured runs and are not presented as such.

Run:

```bash
python figure04_chinchilla_frontier.py
```

Outputs: PNG, SVG, and CSV numerical table.
