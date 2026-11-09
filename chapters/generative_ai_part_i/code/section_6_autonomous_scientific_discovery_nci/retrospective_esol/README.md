# Chapter 6 autonomous scientific discovery — measured-assay addendum

This experiment uses 1,128 **experimentally measured** Delaney/ESOL aqueous solubilities from the publicly available `deepchem/deepchem/datasets/delaney-processed.csv`, originally published by Delaney (2004), DOI 10.1021/ci034243x. The deterministic 902/226 train/test split uses xorshift32 Fisher–Yates with seed 20261109.

A ridge regressor (five molecular descriptors, regularization 10) ranks 226 held-out compounds without looking at the experimental solubility labels. The high-solubility threshold (-1.6 log mol/L) is fixed using the training-set 75th percentile. In its top 20, the fitted model retrospectively identifies 14 experimentally high-solubility records (versus random expectation 5.22); the separate **precomputed** published ESOL prediction column identifies 17. Model test RMSE 1.180 log units, published ESOL prediction RMSE 0.876.

All results were calculated against actual public measured labels obtained from the GitHub dataset via a connected data source; aggregated decile results and full scientific figures are in the Chapter 6 Pass 25 package. Source data retrieval is required for a clean-checkout independent rerun.

**Do not claim autonomous laboratory validation.** These are previously measured, existing compounds, not generated and newly assayed candidates; the 300 novel NCI-derived structure edits from Pass 24 are still without experimental potency or solubility measurements. This is one random split, not scaffold-disjoint or prospective. No lab equipment, robotic experimentation, biological measurements or scientific discovery was performed by the assistant. Unit remains **PARTIAL**, whole-chapter certification remains **HOLD**.

`python reproduce_esol.py --csv delaney-processed.csv` generates full per-compound test predictions and recomputable metrics.