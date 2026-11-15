# Chapter 6 — Pass 58: current ten-scope scientific register and executed release-gate remediation

**Date:** 2026-11-14. **Governing standard:** proportional Master QA R1–R10 for authentic observed data, actual executed models/outputs, scientific figures, and committed code. Scientific and manuscript-production decisions remain separate.

## Live scientific register verification
The default-branch register `qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv` was fetched and parsed as actual CSV, preserving quoted fields. GitHub blob SHA-1: `872605c5bf376ebb30520999ef323cf9fad1aa2b`.

**Recorded scoped scientific certificates: 10, exactly CH6-D01…CH6-D10. Nine Level A; one Level B (D04).** Each row has ten recorded R1–R10 `PASS` values, `SCIENTIFIC_CERTIFIED`, `PRODUCTION_HOLD`, and an explicit stated limitation. These are records of prior scoped certification reviews; verifying the CSV does **not** replace re-execution of all ten scientific models.

The two newest documented scopes are: **D09**, a limited NCI molecular model/fingerprint demonstration with poor convergence and no laboratory or pharmacological validation; and **D10**, a retrospective real measured Delaney ESOL solubility-screening/ranking experiment on **preexisting assayed compounds**, not newly generated or experimentally synthesized molecules. The current register explicitly forbids stronger interpretations. Neither upgrades the original fourteen parent applications to completed approval.

## A concrete defect found and repaired
The live scientific validator expected ten scoped records while `qa/check_ch6_release_gate.py`, its tests, `qa/test_scoped_certification_validator.py`, and `RELEASE_MANIFEST.md` still used the outdated eight-record accounting.

- Revised the release gate to require at least the current **ten** scoped records, **unique and consecutive** `CH6-D01…Dnn` IDs, and truthful per-scope R1–R10/production-hold fields. The minimum prevents a subsequently deleted row from being silently treated as acceptable; the contiguous rule rejects ID gaps. Adding future genuine certificates does not require editing a fixed upper limit.
- Revised release-gate test fixtures to ten records and added regression tests for **deleted certificates** and **gapped IDs**.
- Synchronized the scoped validator regression assertions to ten (9A/1B), the release manifest, and the GitHub workflow's current-certificates label.
- **No scientific scope was promoted, and no parent application was marked fully certified.**

## Executed regression evidence
The two updated release-gate Python files were reconstructed locally from previously archived original GitHub source and modified with the same patch as committed. A local Git-blob SHA-1 check matched the **exact remote current source**:

- `qa/check_ch6_release_gate.py`: `8aff09cfb2711d8ee76a43c3e4136217ddc745af`.
- `qa/test_ch6_release_gate.py`: `b31abd6b46cffc8dd7b94604724c90eee91a30b8`.

Executed:

```bash
python -m unittest discover -s qa -p 'test_ch6_release_gate.py' -v
```

Result: **7 tests executed, 7 passed, 0 failures**, including negative tests for absent files, wrong Figure 6.20 domain, absent caption, removed certificate, and noncontiguous certificate IDs. The one passing fixture is **structural-only** and returns `STRUCTURAL_GATE_PASS_REVIEW_REQUIRED`, never scientific completion.

The separate complete `validate_scoped_ch6_certifications.py` with all its original Yeh, Brownlee, cytology, and ESOL input files was **not** independently re-executed in this pass; CI is configured to run it. This report does not claim a confirmed GitHub Actions result. No scientific model was freshly trained.

## Unresolved Master QA release gates
1. Exact latest current chapter `.tex` and `.pdf` are not present in the accessible Chapter 6 repository root. The user-provided Pass 37 note is source evidence of earlier Figure 6.20 *placement*, not verification of its current *dataset/generator/asset* or latest manuscript build.
2. Figure 6.20's industrial-defect source-to-figure/caption linkage remains unverified; do not substitute the colliding digits `figure_20.png`.
3. Real multi-scene BDD100K generation plus actual downstream independent full-scene perception evidence is not established for synthetic-driving improvement claims.
4. No independently measured physical results on newly **generated** concrete formulations; retained computational Yeh prediction may be scientifically valid with its present narrow scope only.
5. The 14 full parent applications still require exact titled-claim and evidence matching, and final production inspection must verify all current figures/tables/section prose/code listings.

**Decision:** Ten **scoped** scientific certification records are internally consistent and the release test suite passes. **Chapter 6 full parent-application scientific and manuscript-production approval remain HOLD.**
