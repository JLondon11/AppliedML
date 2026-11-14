# Chapter 6 — Pass 51: certification-register validator correction and Figure 6.20 gate

## Verified repository finding

The existing validator `qa/validate_scoped_ch6_certifications.py` contained an internal contradiction:

- `expected={'CH6-D01','CH6-D02','CH6-D03','CH6-D04'}` (four IDs)
- but `len(rows)!=3` caused any correctly formed four-row register to fail before scientific checks.

Corrected the length gate to `len(rows)!=len(expected)`. Added `qa/test_scoped_certification_validator.py` for an explicit four-certificate / three-Level-A-and-one-Level-B / production-HOLD assertion. Source change does not add, remove, or promote a scientific certificate.

## Independent data consistency cross-check

The GitHub-hosted `CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv` contains exactly four distinct IDs: D01, D02, D03 and D04. Each is marked `SCIENTIFIC_CERTIFIED` and `PRODUCTION_HOLD`. The GitHub-hosted `brownlee_original_loo_replay_20261113.csv` has 21 held-out plant records, and independently calculated values were:

| Measurement | Verified |
|---|---:|
| GP RMSE | 4.118931861605197 |
| Ridge RMSE | 3.647664958998736 |
| Mean-control RMSE | 10.422811520890129 |
| Nominal 90% GP coverage count | 19 / 21 |

The complete Python validator and the new unit test have been committed. This check directly recomputed the register identity, Brownlee errors, and coverage from GitHub CSV, but **does not claim an independent execution of the committed Python unit test**.

## Current Figure 6.20 evidence gate

The existing `quality_assurance/chapter6_evidence_manifest.json` states:
- printed Figure 6.20 has a reported **industrial-defect** subject;
- legacy `figure_generation/pass12/real_data_figures.py` generates a **handwritten-digit** robustness plot named `figure_20.png`;
- the candidate-to-manuscript mapping is **unverified**, with output checksum `UNVERIFIED`;
- the chapter's current production figure and exact manuscript LaTeX/PDF are unavailable to inspect.

**This is not proof that the wrong digit image was inserted.** It means source-to-figure provenance for the printed industrial-defect figure cannot be certified merely from a colliding filename or a previous placement statement. The separate, genuinely executed NEU defect benchmark (D04) is scientific evidence about real steel defect classification augmentation, but its source outputs cannot be silently relabeled as Figure 6.20 either.

## Release decision

- Four scoped scientific certificates retained: Level A D01–D03 and Level B D04.
- **Zero full parent Chapter 6 applications are newly certified by this correction.**
- Actual current chapter PDF/TeX and exact Figure 6.20 image, caption, plotted measurements, and source hashes still require a direct matched inspection.
- Production approval remains **HOLD**. Scientific R1–R10 is proportional to specific honest experimental scopes, never a substitute for manuscript integration or code delivery.
