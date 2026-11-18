# Chapter 6 certification registers: scoped science versus the 14 parent studies

**Current as inspected 2026-11-18.** Do not treat provisional labels as certifications.

## Two different inventories

1. `qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv` contains **12** separately registered scientific *experimental scopes*, CH6-D01 through CH6-D12 (**11 Level A and 1 Level B**). Each record reports PASS for R1–R10 **within that specific narrow experiment** and is tagged `SCIENTIFIC_CERTIFIED` and `PRODUCTION_HOLD`. These are not 12 of the chapter's 14 complete applications. Some study boundaries explicitly deny deployed clinical/industrial/dynamic systems, physical casting, full-scene driving, prospective molecular discovery and broad Stable Diffusion evidence.
2. `CERTIFICATION_REGISTER.csv` at the Chapter 6 root is an **older provisional/unmapped record**, with *five* historical rows: industrial defect, driving, concrete, digits and VAE/PCA. Its first four entries are NOT_ASSESSED, and the VAE/PCA row retains PARTIAL on R8/R9 for an earlier delivery/production context. This legacy CSV cannot be substituted as the live exhaustive 14-application inventory, nor used to revoke the narrow empirical D02 certificate without a review of which exact scope and gate its PARTIAL labels describe.

The actual last audited manuscript contained **14 parent case-study/application units**. The exact current full titles/section IDs, figure placements and 14 case-study source-output associations **cannot be reconstructed safely from these two CSV files alone**. They must be extracted from and audited against the current complete manuscript.

## The gating rules

- For a *scoped experiment*, report exactly the narrow scientific claim and its true original-data outputs, not a broader parent title.
- For a *named parent application*, require the exact manuscript title, dataset/task match, an actually run model, source-derived output and figure, error/reference checks appropriate to B or C, complete committed code, and a distinct R1–R10 decision. An A demonstration is not the default parent certification.
- `qa/verify_ch6_parent_science.py` now requires exactly 14 actual parent records in the compiled-manuscript production inventory, full R1–R10 assessments, documented justified N/A only for nonfundamental requirements, task-specific source/evaluation/output files and at least one reviewed original-data result figure. It distinguishes structural evidence-record completeness from independent experimental scientific acceptance.
- `qa/check_ch6_release_gate.py` now **fails closed** if those parent records are missing or incomplete. Neither a perfect software test run nor 12 existing scoped certificates grants Chapter 6 final scientific/production release.
- `qa/verify_ch6_production_inventory.py` independently verifies the historical 48-figure / 17-table / 11-listing / 14-parent inventory and image/evidence paths; `qa/verify_tex_figure_bindings.py` compares the actual compiled-source caption, label and graphics reference. Both remain necessary for production clearance, plus direct PDF inspection.

## Independent evidence limits still open

- Figure 6.20's *printed placement* was reported in the Pass 37 handoff as satisfactory. Current final industrial-defect figure generator, exact actual asset and caption linkage are not directly verified; a candidate digits `figure_20.png` is **not** source evidence for an industrial-defect figure.
- New formulation proposals in the Yeh concrete experiment have **zero independently measured casts/tests**. Predictions for those designs are numerical proposals, not measured strengths.
- The Middlebury one-scene photo CVAE is not a BDD100K multi-scene driving generation-and-downstream-perception benchmark.
- Retrospective ESOL rankings of previously assayed compounds are not autonomous laboratory discovery or prospective measurement.
- The final current Chapter 6 LaTeX/PDF and directly compiled 14-application inventory remain absent from the accessible canonical repository root.

**Decision:** 12 *limited* source-backed scientific certificates remain recorded; **0 additional full parent certifications issued by this reconciliation**; **full Master QA release remains HOLD**.
