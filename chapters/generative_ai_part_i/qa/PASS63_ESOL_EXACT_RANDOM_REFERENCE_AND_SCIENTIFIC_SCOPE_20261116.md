# Chapter 6 — Pass 63: exact random-ranking experiment and scientific-production remediation

**Date:** 2026-11-16. **Policy:** Book-wide proportional R1–R10, source-linked real experimental evidence and publication figures, separate scientifically certified scoped demonstrations from uncategorically approved parent applications.

## Newly independently checked source observations

Read the actual committed original ESOL held-out file `code/section_6_autonomous_scientific_discovery_nci/retrospective_esol/verified_original_ESOL_226_test_predictions.csv` (Git blob SHA-1 `710f647e4f200664b88fe18496e5a9717ddd46b3`). It contains **226 unique preexisting observed compound solubility assays**. Precisely **59/226** original measured labels meet the training-derived high-solubility criterion `measured_logS >= -1.6`. The actual recorded score orderings identify **14/20** high-solubility measured hits for the fitted ridge ranking and **17/20** for the precomputed published ESOL-score column. The latter's upstream training independence cannot be verified and must not be misrepresented as a matched independently trained comparator.

## Exact finite-population null (verified, no Monte Carlo or synthetic observations)

Under the explicit null hypothesis of **uniform random selection without replacement of 20 from these 226 observed test assays**, the number of hits X is hypergeometric with population N=226, K=59 observed successes, sample n=20:

`P(X=k) = choose(59,k) * choose(167,20-k) / choose(226,20)`.

Its sum over k=0,...,20 is 1 (numeric tolerance), expected count **20 × 59/226 = 5.221238938** and central 95% integer quantile range **2–9 hits**.

| Genuine held-out source-ranking result | Measured precision@20 | Enrichment vs 59/226 prevalence | Exact uniform-random-selection tail |
|---|---:|---:|---:|
| Locally fitted ridge (902 original training assays) | 14/20 = 70% | 2.6814× | P(X ≥ 14) = **1.96342660784×10⁻⁵** |
| Precomputed published ESOL score (original source column) | 17/20 = 85% | 3.2559× | P(X ≥ 17) = **1.05679984799×10⁻⁸** |

Both tails were independently evaluated with finite combinations and cross-checked against SciPy's hypergeometric survival function; numerical differences were below floating-point rounding. These exact tails support **nonrandom retrospective enrichment for the fixed ranking and fixed historical assay set**. They do **not** measure probability of clinical/chemical/engineering success, out-of-sample robustness across repeated train/test splits, a causal generative-model advantage, or discovery of an untested molecule. The fixed cutoff and original single model run are important limits. The source's published ESOL column is not a fair independent model-training comparison.

## New code, outputs and original-data scientific figure committed

All source and evidence is organized under:
`chapters/generative_ai_part_i/code/section_6_autonomous_scientific_discovery_nci/retrospective_esol/`.

- `exact_random_selection_reference.py` — standard-library exact hypergeometric random null and per-model ranking, records validated from the actual 226 original assay rows.
- `verified_original_ESOL_random_null.json` — retained numeric null, ranking effect sizes and explicit inferential limitations, derived from the real source file.
- `plot_exact_random_ranking_tail.js` — original-data, deterministic figure generation from the verified finite-population experiment.
- `figures/esol_exact_random_ranking_tail.svg` — readable annotated vector figure of the exact survival curve with actual ridge/published-score outcomes, restrained design instead of wide bar charts.
- `README.md` — revised scientific interpretation and usage.
- `.github/workflows/chapter6_scoped_level_a_certification.yml` — now includes exact-null rerun, numerical source comparison and SVG regeneration-drift guard.

**Execution honesty:** The exact finite-population computation was executed against parsed authentic GitHub CSV evidence in the audit and independently cross-checked numerically in Python against SciPy. The new committed Python file was source-compiled locally; **a full new Python run of that committed file against a local copy of all 226 records was not separately executed in this pass**. GitHub Actions automation for that full check was committed, but a completed CI job has not been verified. No new molecule was trained, generated, physically synthesized or assayed by this pass.

## Certification decision and current production status

This is new support for CH6-D10 **Level A scoped retrospective measured-solubility ranking**, especially R5/R6 reference integrity and R8 measured scientific figure, with no scope promotion to broader **autonomous scientific discovery**. The authoritative Chapter 6 register currently contains **12 scoped certificates (11 A, 1 B)**; these are not 12 of 14 certified parent applications.

**Production remains HOLD**: the accessible GitHub Chapter 6 root has no current complete committed TeX/PDF release. The historical Figure 6.20 placement note cannot replace inspecting the actual industrial-defect image/caption and generator. No genuine BDD100K multi-scene downstream perception benefit demonstration or signed physical concrete tests for *new generated recipes* was presented. The historical manuscript baseline of 48 figures, 17 tables, 11 numbered listings and 14 parent applications still requires complete current direct layout and evidence mapping before release.
