# Chapter 6 — Pass 56: executed QA tests and exact-source package integrity

Date: 2026-11-14. Repository: `JLondon11/AppliedML`.

## Executed test results (not merely source inspection)

Reconstructed each test script and its target implementation **byte-for-byte** from GitHub's default branch source and verified local Git blob SHA-1 against the GitHub fetched file SHA before running.

| Executed suite | Implementation blob SHA-1 | Test file blob SHA-1 | Result |
|---|---|---|---|
| `figure_generation/test_verify_source_provenance.py` | `234dfec872fa4a87effccca089676387c193d908` | `3ebebf62eabd5c22b5677f1e13b41e1011a78654` | **8/8 PASS**, 0 failures |
| `qa/test_ch6_release_gate.py` | `f8ba6d9dfe6860aca075cd4ff7a4993469bc4b88` | `04b74c5214c7e82f764e872f71708945d1c169c1` | **5/5 PASS**, 0 failures |

The five release-gate tests use explicitly artificial fixtures and verify code behavior, **not** any experiment's scientific validity. Test functions correctly block missing manuscript artifacts, missing Figure 6.20 caption, domain mismatch, and missing provenance; the positive fixture returns `STRUCTURAL_GATE_PASS_REVIEW_REQUIRED` rather than scientific approval.

## Authentic retained experiment-package integrity (not new model training)

The exact GitHub-source evidence ZIP `Chapter6_Exact_GitHub_Source_Executed_Evidence.zip` was opened directly and checked against its retained `HEAD_RUN_MANIFEST.json`. The actual committed provenance script was run on all extracted files with that manifest. It returned **8 passed checks, 0 failures** (source Git blob identity, manifest source identity, and six file SHA-256 entries). Separately the ZIP's six listed files had **6/6 SHA-256 matches**, and the source Git blob identity agreed with the manifest (`666cac09a9ec5b127f2ba7d89f109c6b10065df4`).

Files checked: exact-source `vae_pca_real_digits.py`, `output/metrics.json`, `output/real_digits_vae_pca_evaluation.png`, `output/test_predictions.csv`, `output/training_history.csv`, `output/vae_checkpoint.pt`. These checks establish ZIP/source integrity, not fresh model training, cross-seed scientific robustness, or manuscript figure mapping.

## Certificate-count drift observed and corrected

While the audit was in progress, the authoritative scoped register advanced from six to **eight** certificates (CH6-D01 through CH6-D08). The new scopes are real Wisconsin cytology feature augmentation (D07) and real-photograph restoration ablation (D08), each restricted to its documented Level A claims. The unchanged universal R1–R10 policy remains proportional. Existing record statuses are **seven Level A / one Level B**, each separately marked `PRODUCTION_HOLD`. This is **not** eight fully certified parent applications.

The following committed files were aligned to the eight-record register:
- `qa/check_ch6_release_gate.py`;
- `qa/test_ch6_release_gate.py`;
- `qa/test_scoped_certification_validator.py`;
- `RELEASE_MANIFEST.md`.

## Important outstanding limits

The independent Python test for the **scoped register** (which requires the complete Yeh dataset and cytology rows from the full repository) was not executed in this pass. The five release-gate regression tests *were* executed against exact GitHub-source code. The complete production release gate was not executed over a local checkout of the entire repository; its current root tree was inspected through GitHub and lacks the current chapter manuscript assets.

Figure 6.20's actual production image/caption and source-to-image mapping remain unverified. Genuine BDD100K multi-scene downstream perception evidence and independent physical testing of newly generated concrete mixes remain outside completed certification.

**Decision:** QA code execution successful, experimental package integrity verified, scoped register synchronized to eight. Full Chapter 6 parent-application scientific/production release remains **HOLD**.
