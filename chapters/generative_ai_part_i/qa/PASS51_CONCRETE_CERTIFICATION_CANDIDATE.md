# Chapter 6 — Pass 51: independent concrete observation audit

**Current disposition: LEVEL A CERTIFICATION CANDIDATE; SCIENTIFIC HOLD pending exact figure/experiment provenance review.** No physical engineering materials claims are approved. Previous four scoped certificates remain unchanged.

## Independent verification completed against committed source data

The original Yeh concrete observations (`Concrete_Data_Yeh.csv`, 1,030 real laboratory rows), the 229 saved original held-out observation/prediction pairs and the 200 saved numerical generated-candidate proposals were inspected via the GitHub connector.

The original saved 229-row predictions were independently rescored from the `strength_measured` and `strength_predicted` columns:
- RMSE **7.416122281335996 MPa**, matching the repository benchmark.
- MAE **5.288476589935078 MPa**, matching the repository benchmark.
- Held-out original record IDs are all distinct (229 rows).
- 86 genuine held-out 28-day observed mixes; 29 meet the model's low-cement/water screening criteria; **0/29** have an *observed* compressive strength of 40 MPa or above.
- 100 saved **generated numerical candidate** records per arm; their `predicted_strength` is **surrogate output, not measured concrete strength**.

## Newly committed independent checks

`qa/audit_concrete_retrospective.py` is a standard-library record audit which cross-checks held-out source IDs and measured values against original Yeh rows; identifies formula leakage into development groups; verifies the benchmark RMSE and MAE; checks observed 28-day eligibility and zero measured target hits; validates generated candidate labeling and counts; and checks the original source Git blob identity.

**Execution status:** The arithmetic portion was independently executed directly on repository source columns. The new full Python audit is committed for repeatable execution in a clean repository checkout, but its full source-row/group-verification execution has **not** yet been observed in this pass. Do not misstate a committed script as a completed full audit.

## Proportional R1–R10 readiness for a strictly retrospective computational unit

| Requirement | Decision | Evidence / remaining gate |
|---|---|---|
| R1 original real observations | PASS | Yeh source CSV, UCI provenance, original Git blob |
| R2 grouped train/validation/test | PARTIAL | Source asserts 645/156/229 split; independent complete split and group check script committed, still awaiting execution |
| R3 model execution | PASS for historical run evidence | Existing generator script, retained predictions and measured records |
| R4 output/prediction trace | PASS | 229 saved actual held-out prediction pairs, 200 saved computational proposals |
| R5 reference | PARTIAL | Training-mean baseline reported in prior benchmark; independent baseline recreation not yet performed |
| R6 errors | PASS | Independently recalculated original held-out RMSE/MAE and negative observed screen |
| R7 domain validation | N/A — Justified for purely offline proposal screening; **not** for physical performance claims | Original physical validation remains HOLD: zero new mixes cast or tested |
| R8 source-linked scientific figures | PARTIAL | `publication_figures.py` exists but exact current manuscript image asset/caption and plot execution unconfirmed |
| R9 complete committed code | PASS for inspected computational protocol | `run.mjs`, source CSV, recorded predictions, figures generator, lab validator and audit script |
| R10 conclusion limits | PASS | Clearly restrict to original-data retrospective screening and computational proposals |

**Certification decision: NOT YET SCIENTIFIC_CERTIFIED.** Independent physical testing is *not required* for a narrowly titled offline computational demonstration, but unresolved R2/R5/R8 must be resolved before a full Level A certificate. Original concrete-mixture generation with actual measured new specimen strength remains separately HOLD regardless of computational status.

This document is a scientific QA record, not production-manuscript prose.
