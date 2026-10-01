# Chapter 1 Code Listing Integration Audit

Authoritative manuscript baseline: Pass20 PDF, Chapter 1.

Hard rule:
- every retained listing has a title and chapter-scoped number;
- every code line is numbered in the final LaTeX;
- every listing is referenced and discussed in surrounding prose;
- specific line ranges are discussed when they materially implement the concept.

## Pass20 findings

Chapter 1 contains 19 retained listings. All 19 are explicitly referenced in prose. Seventeen already include useful line-specific discussion in Pass20 and should preserve that discussion during reconstruction.

| Listing | Pass20 integration status | Action |
|---|---|---|
| 1.1 AdamW update | line-specific discussion present | preserve |
| 1.2 L-BFGS vs SGD-family solvers | line-specific discussion present | preserve |
| 1.3 CNN optimizer memory/accuracy loop | line-specific discussion present; code body requires repaired runnable implementation from Publication Pass A | preserve discussion + repaired code |
| 1.4 AdamW-to-L-BFGS PINN strategy | lines 5-10 and 12-18 discussed | preserve |
| 1.5 PPO clipped surrogate | line 8 discussed | preserve |
| 1.6 Bearing windowing/spectral features | lines 4 and 10-12 discussed | preserve |
| 1.7 Expected calibration error | lines 10-11 discussed | preserve |
| 1.8 Cost-sensitive fraud threshold | lines 8-9 discussed | preserve |
| 1.9 BEV bounding-box IoU | listing is referenced and conceptually discussed, but no specific line is cited | revise prose to discuss intersection-coordinate construction, nonnegative overlap clamp, and union normalization by exact line ranges |
| 1.10 Latitude-weighted RMSE | line 4 discussed | preserve |
| 1.11 INT8 quantization | lines 4-5 discussed | preserve |
| 1.12 Pinball loss | line 5 discussed | preserve |
| 1.13 PHM08 prognostic score | lines 6 and 8 discussed | preserve |
| 1.14 NDVI | line 5 discussed | preserve |
| 1.15 RandomizedSearchCV | line 6 discussed | preserve |
| 1.16 Optuna define-by-run | lines 4-5 discussed | preserve |
| 1.17 Multi-objective HPO | line 2 discussed | preserve |
| 1.18 Median-pruning HPO | line 6 discussed | preserve |
| 1.19 Tracked medical-imaging HPO | listing is introduced but lacks specific line discussion | revise prose to identify exact lines recording dataset/split/seed/search-space/run metadata and explain why each is required for reproducibility |

## Final reconstruction requirement

The final Chapter 1 source must use the shared `BookCode` listing style so line numbers are shown for every code line. Listing 1.9 and Listing 1.19 must gain explicit line-range discussion during reconstruction. Any code edits that change line positions require the surrounding prose references to be updated accordingly.
