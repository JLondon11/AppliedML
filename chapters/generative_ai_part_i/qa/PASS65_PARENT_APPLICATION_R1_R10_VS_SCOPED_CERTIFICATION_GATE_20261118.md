# Chapter 6 — Pass 65: actual parent-application scientific Master QA release remediation

**Date:** 2026-11-18  
**Standard:** `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md`, proportional R1–R10 with authentic original data, actually executed task-specific models, retained genuine outputs, complete committed code, scientifically relevant figures, and honest conclusions. Separate scientific certification from the compiled-manuscript production check.

## Repository review and critical distinction

The latest accessible Chapter 6 root still has **no current canonical `manuscript/` TeX/PDF folder**. `Proceed Chapter 6.txt` is a previous production *handoff claim* indicating Figure 6.20 placement on page 67, not the actual current final image and manuscript to inspect. The legacy `quality_assurance/chapter6_evidence_manifest.json` explicitly marks the figure-to-source mapping unverified.

The authoritative `qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv` records **twelve experimental scopes** (eleven A, one B), all `PRODUCTION_HOLD`. Those are scoped genuine-data scientific certificates, **not 12 of the 14 complete named parent studies certified**. The older chapter-root `CERTIFICATION_REGISTER.csv` contains **five historical provisional/unmapped** entries (including an older VAE/PCA R8/R9 partial status); it is **not a complete parent-application certificate register**. This classification ambiguity is resolved in `qa/SCIENTIFIC_VS_PARENT_CERTIFICATION_CROSSWALK.md` without quietly overwriting either source.

## Specific release-gate defect fixed

The prior full production gate inspected the scientific *scoped* register and checked counts/links in a 14-parent production inventory, but it **did not independently require that the 14 parent applications actually have full scientific R1–R10 evidence and decisions at their own title/claim breadth**. A complete-looking production inventory could therefore be structurally accepted while one or more original applications remained SCIENTIFIC_HOLD.

Added `qa/verify_ch6_parent_science.py`. The gate now requires **fourteen** real manuscript-titled parent applications with independent full-scope `SCIENTIFIC_CERTIFIED` decisions, Level B task-matched (or Level C broad/research-grade) evidence, a complete R1–R10 evaluation, source documents for actual data/algorithm/results/quantitative evaluation/review, a reviewed original-experimental figure, an exact 40-hex source-code commit, and verified title-to-scientific-claim alignment. The fundamental six empirical requirements R1/R3/R4/R8/R9/R10 cannot be N/A; an actually inapplicable R2/R5/R6/R7 requires substantive rationale. Code-only checks do not certify any experiment.

The integrated `qa/check_ch6_release_gate.py` now adds **`all_parent_scientific_scopes`** as a mandatory release check, independently of the existing twelve demonstration certificates, the full figure/table/listing inventory, manuscript/PDF, Figure 6.20 mapping and LaTeX asset checks. Even fully populated records receive `PARENT_SCOPE_RECORDS_COMPLETE_INDEPENDENT_REVIEW_REQUIRED` as a *structural* review state, not an automatic scientific truth determination. Real independent scientific review remains compulsory.

## Executed QA software tests

Reran the retained Pass 64 original-source software QA archive (34/34 PASS). Added a separately implemented 12-case parent-science fixture suite, including detection of 12-not-14 studies, narrow Level A substituted for a parent B/C study, real output missing, scientific HOLD, title mismatch, fundamental R8 N/A, unjustified R2 N/A, no source commit, invalid figure linkage and duplicated IDs. Added integrated parent-science release test and then three malformed-input fail-closed tests (unhashable parent IDs, corrupt figure IDs, nonstring verdict).

Combined local software verification after these changes: **50 unit tests executed, 50 passed, 0 failed**. The previously archived production/figure test implementation versions and the integrated release-gate patch have Git blob matches to the current corresponding repository versions. The new parent-checker source was mirrored locally for execution and code-reviewed; its exact text differs in nonfunctional formatting from the newly committed GitHub version, so this is **not represented as a byte-for-byte execution of the newly committed GitHub checker**. Its CI check was added but a successful completed GitHub Actions run has not been independently observed.

The new tests are **synthetic software fixtures for correct behavior**, not scientific datasets, generated physical measurements, figures offered for print, or evidence that any real parent case study has passed R1–R10.

## Deliverables in GitHub

- `qa/verify_ch6_parent_science.py` — fail-closed 14-application, full R1–R10 task-claim release evidence checker.
- `qa/test_verify_ch6_parent_science.py` — fifteen software regression cases including malformed records.
- `qa/check_ch6_release_gate.py`, `qa/test_ch6_release_gate.py` — integrated parent-science gate and failure-case test.
- `.github/workflows/chapter6_scoped_level_a_certification.yml` — runs the new parent-scope tests and reports incomplete parent evidence.
- `quality_assurance/PRODUCTION_INVENTORY_SCHEMA.md` — all mandatory exact parent-evidence fields, example with NOT VERIFIED default.
- `qa/SCIENTIFIC_VS_PARENT_CERTIFICATION_CROSSWALK.md` — legacy five-row provisional versus current twelve scoped versus fourteen required complete parent applications.

## Unresolved actual scientific and production requirements

1. **Final current manuscript:** Exact compiled current PDF, full LaTeX, every figure source/caption, corresponding original data assets, cross-references, section text, 14 application titles, and true current inventory remain absent from the canonical accessible repository. The prior Figure 6.20 placement statement is not the full image/data provenance verification.
2. **Driving:** A genuine representative multi-scene driving-generation and independent held-out downstream full-scene perception experiment is needed before claims of synthetic-driving perception gains; Middlebury one-scene patch generation does not suffice.
3. **Concrete:** Computational numerical mixture proposals have zero independent physical tests of new casts. Do not claim experimentally established physical strength or candidate deployment.
4. **Autonomous discovery:** Real ESOL solubility retrospective ranks previously measured compounds. No novel generated compound synthesis/assay/closed-loop lab work is certified.
5. **All fourteen parents:** No claim of full parent application scientific completion without the **exact manuscript-titled** source-linked 14-record, B/C-level R1–R10 evidence ledger and independent assessment. This new gate will block premature final release.

**Decision:** Additional Master QA science-to-production enforcement **implemented and tested as software**. Twelve narrow scientific certificates retained; no newly certified broad parent studies; **full Chapter 6 scientific and production approval remains HOLD**.
