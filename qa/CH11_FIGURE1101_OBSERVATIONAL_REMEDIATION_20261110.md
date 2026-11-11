# Chapter 11 Figure 11.1 — scientific and manuscript QA, 2026-11-10

Original Chapter 11 integrated source: `Chapter11_Figure1122_RealData_Integrated_20261110.zip`, `11_Scientific_AI/Scientific_AI_InProgress.tex`.

## Confirmed flaw and correction
The original Figure 11.1 contained illustrative numerical traces, a fabricated feature array, unverified scatterplot and performance band. Its caption called them "measured", "learned", and "validated" despite no dataset-specific model execution. This is unacceptable for an empirical scientific figure.

The revised publication figure is generated from actual NGDC annual sunspot records distributed by statsmodels; 309 annual records, 1700–2008. Twelve-observation windows are separated chronologically into 178 training (targets 1712–1889), 60 validation (1890–1949) and 59 test observations (1950–2008). Input normalization and PCA are fitted on the training windows only. Five Ridge penalties (0.1, 1, 10, 100, 1000) are compared by validation RMSE; alpha=10 wins. Measured held-out RMSE=19.382488684016636 and persistence baseline RMSE=33.17500590729642. The four panels show authentic observed series, genuine standardized training windows, projected PCA coordinates, and actual held-out predictions.

This is a genuine scientific data-to-prediction pipeline **example**, not evidence for the full range of scientific application domains. The Figure 11.1 and Figure 11.3 use the same sunspot dataset for different instructional purposes, creating a possible redundancy that should be resolved during final book-wide figure-balance review.

## Correctly integrated manuscript
The first chapter paragraph and caption now discuss this specific real dataset and its measured outcomes. The original label `fig:sciai-ecosystem` and Figure 11.1 numbering are preserved. An unused manuscript production-placeholder macro has been removed.

Final delivered package: `Chapter11_MasterQA_Figure1101_20261110.zip` (chapter LaTeX, PDF, all figures, source generator, measured output JSON). After a clean rebuild, two consecutive full pdflatex passes completed. The final PDF is **159 pages**, with zero undefined citation/reference messages and figure 11.1 on PDF page 7 (printed page 2). Page 7 was visually inspected.

PDF SHA256: `2baa8801316835877fff809700bd139d98201dae9ed902dbc3992ab4ebfa60c4`.
New Figure 11.1 PNG SHA256: `5c8de50628271c23292fbae459287d23b2a3f94710f97bff2f7592593775bb70`.
Local figure generator SHA256: `5890a2be4d54046682dc92a11722c079a3d7855585c3c07010ec6e3919b60bed`.

Source code committed: `chapters/scientific_ai/code/figure_11_01_observed_sunspots.py`.

## Master QA remaining HOLD
The new figure is **empirically valid in its stated scope**, but 14 previously identified named Chapter 11 application/case-study units still lack fully supported model-specific result figures. The original MoNuSeg nuclear segmentation and paired cancer-cell classification benchmark remains incomplete. The compiled complete chapter and assets are delivered in ZIP but are not yet verified committed into canonical GitHub release layout; GitHub Actions clean-checkout run of this newly committed generator not verified. Do not mark the full chapter or entire book complete.
