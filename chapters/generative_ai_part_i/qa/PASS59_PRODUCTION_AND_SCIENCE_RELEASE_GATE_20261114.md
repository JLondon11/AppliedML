# Chapter 6 — Pass 59: scientific and production remediation

**Standard:** repository-root `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md`, with proportional R1–R10, authentic data, task-specific executed models, original-data publication figures, traceable code, truthful scope, and a **separate production gate**.

## Current scientific certification scope
The current authoritative Chapter 6 register (`qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv`, Git blob SHA-1 `0c8bd7d231fae6d1ed4373f5075cc5a65a8eec82`) lists **12 scoped** scientific certificates: **11 Level A, 1 Level B**, IDs D01–D12. This count does **not** describe 12 of 14 parent applications scientifically complete. All 12 scoped records retain `PRODUCTION_HOLD`. No new empirical certification was issued in Pass 59 and no model-training execution occurred.

## Actual production gap identified and corrected
The previous release gate could return structural clearance after inspecting only Figure 6.20 and finding any LaTeX/PDF filenames. That was insufficient for a chapter whose last compiled audited baseline reported **48 figures, 17 tables, 11 numbered code listings and 14 empirical parent case studies/applications**.

The new committed `qa/verify_ch6_production_inventory.py` and integration in `qa/check_ch6_release_gate.py` now fail closed unless:
- there is an actual, current production inventory linked to a compiled PDF checksum and direct manuscript review;
- all 48 baseline figure slots have distinct IDs, actual asset paths/hash, meaningful captions and scientific review;
- every *experimental* figure has an actual committed generator, original-results path and explicitly checked dataset provenance;
- all 14 empirical parent applications have true current titles, scope decisions, at least one source-linked experimental result figure, and evidence for any scientific approval;
- all 17 baseline tables are individually assessed for substantive relevance and cited source/derivation;
- all 11 code listings trace to complete committed implementations and code review;
- reported Figure 6.20's evidence and actual caption match the industrial-defect subject, instead of merely colliding with a legacy digits filename.

These counts refer to the last independently compiled 109-page baseline. Any actual revised current manuscript with a different substantive figure/table/listing count requires a new independent compiled inventory and updated corresponding baseline—not silent count waiver.

## Executed software QA — exact GitHub source
An initial integrated run caught an actual defect in the previously committed negative-test fixture. The 12-certificate register had been mistakenly represented with 10 rows and 12 requirement fields, so the test could pass vacuously. Both the fixture and its regression coverage were corrected, including a positive check of valid certificate baseline.

The final **21/21 Python regression tests passed, 0 failures**, with model/fixture-only scope. The executed files were independently reconstructed and their Git blob SHA-1 values compared exactly with the live GitHub version:

| Executed committed source | Git blob SHA-1 |
|---|---|
| `qa/check_ch6_release_gate.py` | `31de5b25a7b691c9e4f3664cde8fcfc911c55033` |
| `qa/test_ch6_release_gate.py` | `1f557841143fd84851d6695c056fc37accc5a5ec` |
| `qa/verify_ch6_production_inventory.py` | `3abd20b9a17ac54d96086961deba72e197235543` |
| `qa/test_ch6_production_inventory.py` | `c8efa054bbd12a2281b6a6c53676dea6c4de0086` |

Ten production-inventory tests and eight integrated release-gate tests were executed, plus three further negative tests for missing table/listing coverage and code source checks; the combined total is 21. The fixture-only tests use explicit non-scientific software-test images and generated metadata; **these are not datasets or scientific outputs used in the textbook**.

The `.github/workflows/chapter6_scoped_level_a_certification.yml` now includes both new suites and release-gate checks. A separate GitHub Actions completion status was **not** independently verified here.

## Unresolved genuine scientific / editorial production work

1. **Current manuscript unavailable** in the accessible Chapter 6 repo root, including the exact PDF/LaTeX, image assets and captions. The archival 109-page Pass 44/45 description cannot certify the present manuscript.
2. **Figure 6.20** remains an unverified source-to-manuscript mapping. The existing candidate manifest describes a *handwritten-digit* figure `figure_20.png` while the reportedly printed figure is an *industrial-defect* illustration. Do not infer an actual substitution until the manuscript figure asset is inspected.
3. **Synthetic driving** still needs genuine multi-scene BDD100K and held-out full-scene downstream perception evaluation before any benefit claim is certified. The certified one-scene Middlebury demonstration does not substitute.
4. **Generated concrete** remains purely computational for new proposed formulas. The Yeh measured retrospective set does not provide actual independent physical cast/strength tests for novel proposals; no such test data have been invented.
5. **Full parent applications, narrative, tables, section introductions, references and all figure placements** need the current manuscript and original saved experiment artifacts before final release decisions.

**Decision:** New source-linked production-coverage checker committed and 21 software tests passing. **12 scoped scientific certificates retained, zero parent applications newly promoted. Full Chapter 6 scientific/production approval remains HOLD**, as required by the updated Master QA standard.
