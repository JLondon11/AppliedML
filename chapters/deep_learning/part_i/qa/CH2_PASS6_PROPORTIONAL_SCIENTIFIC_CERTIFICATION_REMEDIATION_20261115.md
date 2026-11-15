# Chapter 2 — Pass 6 scientific certification and reproducible-release gate

Date: 2026-11-15. Governing standard: `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md`, proportional R1–R10.

## Exact source inspected
The actual user-supplied `Chapter2_Pass5_Production_Code_Figure_Audit_20261115.zip` was extracted and inspected, not merely inferred from previous GitHub reports. It contains the revised Deep Learning Part I LaTeX, 118-page PDF, actual test predictions, 35 referenced figures, and executable numerical validators.

## Confirmed executable packaging defect
The existing `qa_code/verify_measured_case_studies.py` pointed to its own `qa_code/results/measured_results.json` and `qa_code/figures/`. Neither path exists in the actual release tree. The first invocation failed with `FileNotFoundError`. Fixed the script to use `Path(__file__).resolve().parents[1]` as the release root, `evidence/measured_results.json` and `figures/`. **The corrected verifier was actually executed and returned PASS** for original dataset digests, source splits, 450 real Digits predictions, 111 real Diabetes predictions, scores, uncertainty diagnostics, and all four result-figure groups.

Two existing original-data verifiers also passed independently: `code/validate_real_digits_nonlinear.py` (original 3/8 image labels, 108 real held-out MLP/logistic predictions, ROC AUC, source hash) and `code/validate_pass4_outputs.py` (original digit image source, exact three 8x8 fixed-kernel response matrices, twelve independent model seeds, each 450 real held-out predictions, seedwise metrics and recorded training histories).

A new `qa_code/verify_ch2_scientific_evidence.py` orchestrates the three independent numeric verifiers and checks complete 21 PNG/PDF/SVG real-output assets, authentic numerical JSON and all 35 image references. **Executed successfully**. The exact outcome is saved in the deliverable as `CH2_SCIENTIFIC_GATE_EXECUTION.json`. A second, self-contained tool `chapters/deep_learning/part_i/qa/verify_ch2_distributed_science.py` has been committed to GitHub to verify the same numerical evidence directly from the distributed ZIP.

## Scientific decision
Five of five units in the revised uploaded manuscript retain **scientific certification only within their defined scopes**: (A) original two-class Digits MLP with training-only PCA; (A) fixed-kernel response computation on authentic Digits pixels; (B) 10-class Digits MLP; (B) observed Diabetes regression MLP versus Ridge, **inferior RMSE and nonconvergence disclosed**; (B) 12 real-data MLP seed runs. Each has genuine observations, model/kernel execution, matching actual saved outputs and a model-specific measured-results figure. The fixed-kernel demonstration is NOT a trained CNN benchmark. Three actual MLP test tasks reuse some source images but must never be counted as independent patient observations.

This pass is **verification and release repair, not new neural-network training**. It does not report improved performance or newly acquired data. No incorrect old two-moons simulated experiment is counted as an empirical certificate. Certification at A/B does not imply research-grade C generalizability.

## Deliverables and holds
- Packaged revised release: `Chapter2_Pass6_Scientific_Certification_Remediation_20261115.zip`, contains complete manuscript, unchanged PDF, original figures, corrected numeric verifier, new consolidated gate, actual executed JSON report.
- PDF remains 118 pages with unchanged SHA256 `ced984f6f1a1b17fc6817b50a1379f15b27b27558d711c479445268e5f2b4e69`.
- ZIP passed CRC integrity inspection.
- Source-level proofing remains separate: full final bibliography review, full chapter print-size visual inspection and canonical GitHub manuscript and binary publication remain unverified.
- GitHub Actions/clean-checkout execution of the newly committed distributed ZIP validator has not been confirmed.

**DECISION: CHAPTER 2 EMPIRICAL SCOPED SCIENCE 5/5 PASS, FINAL PUBLICATION/MASTER QA HOLD.** The old numerical verifier defect is remediated in the delivered ZIP. The complete revised LaTeX and binary figure assets are not claimed committed to GitHub.
