# Scientific AI Chapter 11 — authentic-data figure and manuscript integration

Date: 2026-11-10.

**Actual manuscript work:** Recovered the full `Scientific_AI_InProgress.tex` and original figure assets from the supplied `All_Chapters_Complete_12.zip`. This pass replaced three original figures *in the real chapter source*, while retaining their original LaTeX labels and numbering. The completed source and binary assets are available in the session deliverable `Scientific_AI_Chapter11_RealData_Integrated_20261110_FINAL.zip`. They have **not** been verified uploaded to GitHub; the complete generator code is committed separately.

## Verified replacements

**Figure 11.3:** The original six-panel conceptual evolution chart contained artificially generated noisy values represented as experimental observations. Replaced with actual National Geophysical Data Center annual sunspot observations (309 records, 1700–2008), empirically measured one-year lag, and genuine one-step Ridge versus persistence held-out predictions. Chronological window split 178/60/59, test RMSE Ridge 19.38, persistence 33.18. Updated caption, explanatory paragraphs, and sunspot dataset source in references. The real-data scientific generator already committed is `chapters/scientific_ai/code/figure_02_scientific_discovery.py`; publication-quality renderer is `chapters/scientific_ai/code/render_verified_figures_02_03.py`. The source chapter's prior label `fig:sciai-evolution` is preserved, even though the figure's content is now a genuine real-data scientific analysis rather than an invented stage-by-stage numerical story.

**Figure 11.8:** Replaced the outdated workflow figure with genuine WDBC five-fold training-only ROC-AUC and actual held-out four-model ROC on 114 untouched cases (569 total cases). Matched caption and quantitative discussion, and corrected Listing 11.3 to remove train/validation leakage from standardizing before folds; the replacement excerpt uses a pipeline for scaling separately inside each fold. Full experiment: `chapters/scientific_ai/code/figure_08_wdbc_baselines.py`.

**Figure 11.11:** Replaced synthetic analytical heat-equation panels with real heated-aluminum rod measurements from eight original thermocouples (168 time samples each), trained a field model on 1,008 observations across six channels, validated on independent CH3 (168), and tested on untouched CH6 (168). Corrected the caption and added multiple introductory scientific background, methods, limitations, measured-error and negative-outcome paragraphs. Full executable code: `chapters/scientific_ai/code/figure_11_pinn_residual.py`. Original lab source: `TariqSawaqed1/Heat-Conduction-Aerospace-Lab`.

## PDF verification

- Actual complete revised chapter successfully compiled twice with pdflatex.
- **158 pages**, PDF SHA-256 `33678a626e56ba1436f10d16e1b73fc1e87f11a549a7e7062d6b43ce1f0b0431`.
- Final second compilation: **zero undefined citation or cross-reference warnings**.
- No duplicate LaTeX labels, original Figure 11.3/11.8/11.11 numbers retained.
- Inspected finished PDF pages **14, 33, 41** for graph/caption/method alignment and placement after introductory discussion. Removed large avoidable whitespace caused by an unbreakable legacy listing wrapper.

## Honest completion scope

This certifies the three **specific integrated replacements at the examined source and output level**, not the scientific or aesthetic quality of every remaining Chapter 11 figure. Other application/case-study figures and text require source-to-data and publication audits. The archive may not be the newest canonical version from other book chats, so merge cautiously. Full edited manuscript and binary figures are in the delivered ZIP, **not verified in GitHub**. Book-wide Master QA remains HOLD.
