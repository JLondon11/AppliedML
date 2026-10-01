# Chapter 1 Frozen Final Content Inventory — Consolidation Target

Authoritative baseline: Pass20 Chapter 1, modified only by accepted publication remediations.

## Final full Applications
1. Predictive Maintenance for Industrial Rotating Machinery
2. Medical Image Classification
3. Financial Fraud Detection Under Extreme Class Imbalance
4. Smart Grid Forecasting and Energy Systems
5. Aerospace Structural Health Monitoring and Turbofan Prognostics

Each retained Application must contain:
- multiple substantive introductory/background paragraphs;
- chapter-specific Problem/Dataset/Model/Method/Evaluation/Results/Error Analysis/Limitations/Deployment or Engineering Implications fields;
- at least one scientifically appropriate, nonredundant figure;
- at least one evidence-based benchmark/comparison table;
- explicit discussion of every figure/table/listing used in the section.

## Final Case Studies
1. Cross-Domain Optimizer Selection
2. Reproducible Hyperparameter Optimization

Both Case Studies must be analytical rather than merely procedural and must include uncertainty, comparison, error/failure analysis, limitations, and research implications.

## Optimization material reclassified as Worked Examples
The former credit-risk, CNN, transformer, PINN, and PPO optimizer “Applications” are retained only as worked examples illustrating optimizer behavior. They do not count as full Applications and do not trigger redundant Application figure/table packages.

## Sections removed from Chapter 1 and assigned to later chapters
- Intelligent Transportation / Autonomous Driving -> Computer Vision Part II / Reinforcement Learning
- Climate and Weather Scientific ML -> Scientific AI
- Edge AI for Industrial IoT -> Systems Engineering and MLOps
- Precision Agriculture / Autonomous Farming -> appropriate CV / Scientific AI / Systems section

These are removed from Chapter 1 because their primary learning objective belongs to later specialized chapters and retaining them would create cross-chapter duplication.

## Final Chapter 1 listing target
Pass20 contains 19 listings. Removing the four moved domain sections removes:
- former Listing 1.9 — BEV bounding-box IoU
- former Listing 1.10 — latitude-weighted RMSE
- former Listing 1.11 — INT8 quantization
- former Listing 1.14 — NDVI

This leaves **15 substantive Chapter 1 listings**, exactly within the required 15–22 range.

Retained listing roles:
1. AdamW update
2. L-BFGS vs SGD-family convex solver comparison
3. CNN optimizer memory/accuracy worked example
4. hybrid AdamW/L-BFGS PINN worked example
5. PPO clipped-surrogate worked example
6. bearing windowing/spectral features
7. expected calibration error
8. cost-sensitive fraud threshold
9. pinball loss for probabilistic forecasting
10. asymmetric PHM prognostic score
11. randomized search with cross-validation
12. Optuna define-by-run search
13. multi-objective HPO
14. median-pruning HPO
15. tracked/reproducible HPO

Final sequential renumbering is deferred until manuscript integration is stable. The complete repository implementations already exist; moved listing implementations are preserved in git history and will move with their destination sections rather than being deleted.

## Acceptance rule
No additional Application, Case Study, figure, table, or listing may be added merely to increase breadth or page count. New material must close a documented scientific or pedagogical gap and must not duplicate later chapters.
