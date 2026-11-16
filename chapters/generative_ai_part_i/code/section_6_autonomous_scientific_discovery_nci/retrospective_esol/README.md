# Chapter 6 autonomous scientific discovery — measured-assay addendum

This experiment uses 1,128 **experimentally measured** Delaney/ESOL aqueous solubilities from the publicly available `deepchem/deepchem/datasets/delaney-processed.csv`, originally published by Delaney (2004), DOI 10.1021/ci034243x. The deterministic 902/226 train/test split uses xorshift32 Fisher–Yates with seed 20261109.

A ridge regressor (five molecular descriptors, regularization 10) ranks 226 held-out compounds without looking at the experimental solubility labels. The high-solubility threshold (-1.6 log mol/L) is fixed using the training-set 75th percentile. In its top 20, the fitted model retrospectively identifies 14 experimentally high-solubility records (versus random expectation 5.22); the separate **precomputed** published ESOL prediction column identifies 17. Model test RMSE 1.180 log units, published ESOL prediction RMSE 0.876.

All results were calculated against actual public measured labels obtained from the GitHub dataset via a connected data source; aggregated decile results and full scientific figures are in the Chapter 6 Pass 25 package. Source data retrieval is required for a clean-checkout independent rerun.

**Do not claim autonomous laboratory validation.** These are previously measured, existing compounds, not generated and newly assayed candidates; the 300 novel NCI-derived structure edits from Pass 24 are still without experimental potency or solubility measurements. This is one random split, not scaffold-disjoint or prospective. No lab equipment, robotic experimentation, biological measurements or scientific discovery was performed by the assistant. Unit remains **PARTIAL**, whole-chapter certification remains **HOLD**.

`python reproduce_esol.py --csv delaney-processed.csv` generates full per-compound test predictions and recomputable metrics.
## Scoped scientific certification — 2026-11-14

The genuine **retrospective original ESOL measured-solubility ranking** is now independently verified as **CH6-D10, Level A SCIENTIFIC_CERTIFIED**, explicitly limited to existing previously measured compounds. The original Delaney/DeepChem data source was retrieved from pinned public commit `108af2176a5f876c2df1df922baadb02d647c826` and the original experiment's xorshift32 split, training-only standardization, five-descriptor ridge solution, held-out predictions and measured top-20 hit counts were recomputed from the 1,128 measured source records.

The reconstructed **226 original held-out assay measurements and model predictions** are now preserved in `verified_original_ESOL_226_test_predictions.csv` with exact summary `verified_original_ESOL_recalculation.json`. Original-source ridge top-20 hits **14**, RMSE **1.179576933**; published precomputed ESOL predictor hits **17**, RMSE **0.875568503**; random-selection expected hits **5.221**. The high-solubility decision threshold **-1.6 log mol/L** was chosen using training labels only. `plot_verified_original_assays.py` generates a per-compound real-label-vs-prediction non-bar figure from the retained original observations.

No new compound was synthesized or assayed, no lab automation took place, and the independently recomputed source **does not certify the broader autonomous scientific discovery application**. Its parent study remains scientific HOLD where new-molecule experimentation is claimed, and the current full Chapter 6 manuscript remains **PRODUCTION_HOLD**.

**Baseline independence qualification:** The published ESOL prediction column is included in the source dataset and was not retrained or independently leakage-audited in this chapter. Its test-row RMSE and top-20 counts are valid descriptive comparisons with measured labels, but must **not** be described as an independently held-out, information-matched architecture benchmark or evidence that the published predictor generalizes better than the locally fitted ridge model. The ridge model's own 902/226 split and training-only coefficient selection are verified independently.

## Pass 63: exact random-selection reference on original measured assays

The full saved `verified_original_ESOL_226_test_predictions.csv` was independently re-parsed; it contains **226 unique previously assayed compounds**, including **59** authentic measured high-solubility labels (threshold −1.6 log mol/L, originally set using training measurements only). The model-score ranking identifies **14/20** measured hits for locally fitted ridge and **17/20** for the precomputed published ESOL score. For uniform random selection of 20 without replacement from these specific measured test compounds, the exact hypergeometric null gives:

- Expected high-solubility hits: **5.22124**, central 95% integer range **2–9**.
- Ridge: **70.0% precision at 20**, enrichment **2.681×** over held-out prevalence, exact random-rank upper tail **1.96343×10⁻⁵**.
- Precomputed published-score ranking: **85.0% precision at 20**, enrichment **3.256×**, exact random-rank upper tail **1.05680×10⁻⁸**.

These finite-population probabilities describe **how unusual the fixed hit counts would be under uniformly random selection**, not independent replicate validation, causal model gain, new compound synthesis, or success on future unseen assays. The published predictor's original training provenance and overlap are not established, so the two model scores **must not be compared as independently trained like-for-like benchmarks**.

Executable standard-library analysis: `exact_random_selection_reference.py`; frozen original-output audit: `verified_original_ESOL_random_null.json`. The data-linked vector figure is `figures/esol_exact_random_ranking_tail.svg`, generated by `plot_exact_random_ranking_tail.js`. Current chapter LaTeX/PDF figure assignment is separately unverified; these are source-linked scientific figure assets for possible editorial integration, not approved current manuscript figures.

This strengthens the existing **CH6-D10 narrow Level A retrospective screening demonstration**, while the broader autonomous laboratory and newly generated molecular discovery application continues to lack prospective tests.
