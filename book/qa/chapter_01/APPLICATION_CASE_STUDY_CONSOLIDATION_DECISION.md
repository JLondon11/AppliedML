# Chapter 1 Application / Case Study Consolidation Decision

Authoritative baseline: Pass20 Chapter 1.

The final Foundations chapter should not function as a miniature version of later specialized chapters. Applications remain only when the domain is being used to teach a distinctly foundational ML question.

## Retain and rebuild in Chapter 1
1. **Predictive Maintenance for Industrial Rotating Machinery**
   - Foundational concepts: supervised feature engineering, time-window construction, temporal leakage prevention, imbalance, calibration, metric choice.
2. **Medical Image Classification**
   - Foundational concepts: train/validation/test design, class imbalance, calibration, generalization, leakage, reproducibility.
3. **Financial Fraud Detection**
   - Foundational concepts: extreme imbalance, cost-sensitive learning, PR-AUC, threshold selection, chronological validation.
4. **Smart Grid Forecasting and Energy Systems**
   - Foundational concepts: regression, temporal validation, covariate shift, loss design, uncertainty and calibration.
5. **Aerospace Structural Health Monitoring and Turbofan Prognostics**
   - Foundational concepts: supervised regression, temporal validation, asymmetric scoring, leakage, model comparison.
6. **Case Study: Reproducible Hyperparameter Optimization**
   - Foundational concepts: search spaces, cross-validation, pruning, multi-objective optimization, reproducibility, repeated-seed uncertainty.
7. **Case Study: Cross-Domain Optimizer Selection**
   - Foundational concepts: optimization geometry, conditioning, convergence, generalization.

## Reclassify as worked examples, not full Applications
- Credit-risk optimizer example.
- CNN optimizer comparison.
- Transformer optimizer comparison.
- PINN optimizer strategy.
- PPO optimizer example.
These are retained only where they illustrate the optimization section and do not trigger the full Application figure/table requirement.

## Remove or move from Chapter 1
- **Intelligent Transportation and Autonomous Driving** → primarily Computer Vision Part II / Reinforcement Learning.
- **Scientific ML for Climate and Weather Forecasting** → primarily Scientific AI.
- **Edge AI for Industrial IoT** → primarily Systems Engineering and MLOps.
- **Precision Agriculture and Autonomous Farming** → move to the chapter where the actual retained method belongs (CV/Scientific AI/Systems); do not keep as a generic Foundations sidebar.

## Rationale
The removed/moved sections are not being discarded because the domains are unimportant. They are removed from Foundations because retaining them here would either:
- duplicate later chapter material,
- require generic rather than chapter-specific figures/tables,
- inflate Chapter 1 without deepening foundational understanding, or
- force superficial architecture discussion that belongs elsewhere.

The final Chapter 1 application set is intentionally smaller but deeper. Every retained full Application/Case Study must have substantial background prose, a chapter-aligned scientific figure, an evidence-based comparison table, and explicit discussion of both.
