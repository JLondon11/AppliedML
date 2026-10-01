# Chapter 1 Application / Case Study Concept-Alignment Audit

Authoritative source: Pass20 PDF, Chapter 1 (pages 1-113).

Publication rule: every retained Application and Case Study must explicitly apply concepts taught in Foundations of Machine Learning. Sections that primarily teach specialized later-chapter material are rewritten to use the domain only as context for foundational concepts, consolidated, moved, or removed.

## Foundational concept spine
Chapter 1 develops:
- formal learning problems, hypothesis spaces, capacity, PAC/uniform convergence;
- optimization for machine learning;
- representation/feature geometry and dimensionality reduction;
- dataset quality, distribution shift, class imbalance, label noise, missingness, and leakage;
- loss/objective design;
- model selection and hyperparameter optimization;
- generalization and regularization;
- experimental methodology, statistical validation, and reproducibility;
- algorithm-family selection;
- deployment, governance, and risk fundamentals.

## Retained section alignment

| Section | Chapter-1 concepts that must be applied | Action |
|---|---|---|
| Optimizer worked examples | optimization geometry, conditioning, convergence, validation | Keep as worked examples, not full Applications |
| Cross-Domain Optimizer Selection | convex optimization, conditioning, validation, generalization | Consolidated Case Study; retain |
| Predictive Maintenance | supervised regression/classification, feature engineering, temporal leakage control, imbalance, calibration and metric selection | Retain only if rewritten around these foundations |
| Aerospace Prognostics | regression formulation, temporal validation, loss/metric choice, leakage, model comparison, uncertainty | Retain as a Foundations Application; specialized sequence architecture detail is minimized |
| Financial Fraud Detection | severe class imbalance, cost-sensitive learning, PR-AUC, calibration, threshold selection, temporal validation | Retain if tied to foundational evaluation/imbalance concepts |
| Smart Grid / Energy Forecasting | regression, covariate shift, temporal validation, loss choice, feature engineering | Retain if focused on foundational methodology rather than power-systems architecture |
| Precision Agriculture | representation/feature engineering, missing data, spatial/temporal splitting, uncertainty and generalization | Retain if focused on foundational ML formulation |
| Healthcare / Medical Risk | dataset shift, calibration, class imbalance, fairness, leakage, reproducibility | Retain if evidence and benchmark protocol are strong |
| Manufacturing / Quality | classification/regression formulation, anomaly detection baselines, data quality, HPO, validation | Retain if foundational |
| Recommender / Customer Modeling | objective design, implicit feedback, train/test leakage, ranking metrics, regularization | Retain only if it stays at foundations level |
| Any section centered on CNN/Transformer/RL/PINN architecture internals | specialized later-chapter content | Move detail to relevant later chapter; retain only foundational framing if needed |

## Acceptance gate
A section fails Chapter 1 alignment if a reader could remove the Foundations-specific discussion and still retain essentially the same section. Every retained section must explicitly explain why the chapter's concepts determine the problem formulation, model choice, validation protocol, metrics, and interpretation.
