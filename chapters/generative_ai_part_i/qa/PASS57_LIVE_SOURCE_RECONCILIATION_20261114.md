# Chapter 6 — Pass 57: live-source reconciliation and next certification gates

Date: 2026-11-14. Repository: `JLondon11/AppliedML`, default branch `main`.

## Sources directly examined
- `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md` (proportional R1–R10 and separate scientific/production decisions)
- `chapters/generative_ai_part_i/qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv`
- `chapters/generative_ai_part_i/qa/PASS56_EXECUTED_QA_20261114.md`
- `chapters/generative_ai_part_i/CERTIFICATION_REGISTER.csv` (legacy, provisional)
- `chapters/generative_ai_part_i/RELEASE_MANIFEST.md`
- `chapters/generative_ai_part_i/quality_assurance/chapter6_evidence_manifest.json`
- GitHub chapter-root directory listing and `qa/` directory listing.

This is a **live-source document and traceability review**, not a new execution of models, figure byte inspection, manuscript PDF inspection or an independent re-adjudication of all eight certificates.

## Reconciled scientific decision
The authoritative scoped register currently contains eight rows CH6-D01 through CH6-D08: seven explicitly limited Level A scientific certificates (D01, D02, D03, D05, D06, D07, D08) and one offline Level B certificate (D04). Each row records R1–R10 as PASS **for its own narrowly defined experimental claim** and `production_status=PRODUCTION_HOLD`. Negative results must be retained verbatim, notably GAN augmentation underperforming matched controls and the digit VAE not demonstrating superiority to PCA.

The legacy root `CERTIFICATION_REGISTER.csv` remains a **provisional historical inventory** with `UNMAPPED-*` unit IDs and NOT_ASSESSED values. Its rows are not equivalent to the newer D01–D08 register and must not overwrite the latter. A future clean-up should explicitly label the older file as historical and introduce a parent-application-to-D-certificate crosswalk based on actual latest manuscript headings.

Archived Chapter 6 counts (14 named applications/case studies; 48 figures, 17 tables, 11 code listings) are from Pass 44, not a validated count for a current manuscript release.

## Blocking production and broad scientific gaps
1. Current chapter GitHub root does not expose the required `manuscript/` folder containing exact latest TeX and PDF; no new PDF/TeX asset was supplied in this turn. Therefore no new visual page or figure-caption verification is asserted.
2. `quality_assurance/chapter6_evidence_manifest.json` explicitly marks Figure 6.20's candidate source as `sklearn-handwritten-digits` while the reported printed topic is industrial defects. `manuscript_mapping_verified=false`, `output_sha256=UNVERIFIED`, and the candidate's execution is unverified **in that manifest**. A filename collision alone is not evidence that the actual printed figure is wrong. Resolve with exact latest TeX, rendered image, caption, experimental asset, code, and SHA-256.
3. The scoped single-Middlebury-scene stereo experiment (D06) does not establish multi-scene BDD100K synthetic driving or downstream perception improvements.
4. Concrete D05 establishes retrospective strength prediction and numerical proposals only. It does not establish measured physical properties of the newly generated concrete mixes. New physical strength claims require independent specimen tests; otherwise narrow chapter language to the verified computational scope.
5. Scoped certifications do not automatically approve fourteen parent applications. A current manuscript-specific parent-unit crosswalk and individual claim/title decisions are required.

## Verified provenance and limits for earlier work
Pass 56 documents source-matched executed QA suites (8/8 provenance unit tests and 5/5 release-gate unit tests) and six artifact SHA-256 checks on the retained VAE evidence ZIP. These results are **reported in the recorded audit**, not newly rerun in Pass 57. The same report explicitly distinguishes artificial test fixtures from experimental data and retains chapter HOLD.

## Release decision
- Eight existing scope-specific entries: **recorded SCIENTIFIC_CERTIFIED**, constrained to their recorded claims; not independently retested in Pass 57.
- Broad parent scientific claims: **NOT CERTIFIED AS A SET**.
- Figure 6.20 exact current manuscript provenance: **NOT VERIFIED**.
- Current Chapter 6 final production: **PRODUCTION_HOLD**.
- Complete Chapter 6: **NOT COMPLETE**.

## Next evidence-first actions
1. Recover and commit exact current Chapter 6 release TeX/PDF, production assets, and their hashes in the canonical release structure.
2. Resolve Figure 6.20 with an original-to-rendered-asset trace and inspect the actual page; do not infer validity from candidate script basename.
3. Crosswalk all current application headings to D01–D08 and enumerate unmatched claim-specific evidence gaps.
4. Execute any still-required real-data task-matched models, retain per-observation predictions and scored outputs, regenerate figures from these outputs, and certify only the corresponding experimental scope.
5. Execute full release gate against the assembled current manuscript and record separate production approval.
