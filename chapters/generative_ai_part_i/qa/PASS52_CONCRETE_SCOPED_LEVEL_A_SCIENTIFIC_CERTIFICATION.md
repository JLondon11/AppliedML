# Chapter 6 — Pass 52: computational concrete scoped Level A scientific certificate

**Decision:** `CH6-D05 SCIENTIFIC_CERTIFIED` only for original-data **retrospective strength prediction and computational formula proposal**. The parent claim of physically validated, high-strength newly generated concrete remains **SCIENTIFIC_HOLD**. Production remains HOLD pending the exact manuscript and figure integration audit.

## Original observations and independently checked provenance

Source: genuine I.-C. Yeh / UCI concrete laboratory strength dataset; 1,030 original material recipes at measured curing ages and observed strengths. Original CSV Git blob `93f829fc160bac648bc84d034574a18b3685a2ef`, already committed at `code/section_6_engineering_concrete_mix_generation/Concrete_Data_Yeh.csv`.

Exact material-recipe grouping: **427 unique formulations**. The deterministic native-JS shuffle and split were independently reconstructed from the original CSV: **277 train groups = 645 rows; 64 validation groups = 156 rows; 86 held-out test groups = 229 rows**. Saved held-out IDs are identical to the independent deterministic split (same order). No test-versus-development formulation group overlap, and all 229 observed strengths, cement, water, curing ages and binder totals agree with the original source rows.

## Authentic executed model outputs and reference

Original `run.mjs` selects k from 3, 5, 9, 15 by genuine validation error; selected five-neighbor predictor is a standardized inverse-distance regression with log-transformed curing age. Real source records and the original experiment's `heldout_test_predictions.csv` and `benchmark_summary.json` are retained and committed.

Independent recomputation of the 229 observed-versus-predicted pairs:
- Held-out **RMSE 7.416122281335996 MPa**, **MAE 5.288476589935078 MPa**.
- Independently reconstructed **training-only mean**: 35.24666666666667 MPa; baseline held-out **RMSE 16.70038733582021 MPa**, **MAE 13.677598253275105 MPa**.
- Of 86 original genuine test specimens at 28 days, 29 satisfy the exploratory low-cement/screen conditions; **0/29** have an observed strength at least 40 MPa. No experimental low-cement high-strength discovery was established.
- Original training-data-conditioned numerical generator produced 300 proposals per arm according to retained `benchmark_summary.json`; 100 saved formula records per arm were checked in `conditional_generated_designs.csv`. These are **model-generated mathematical proposals**, **not newly tested physical specimens**. No 28-day strengths have been observed for any generated formula.

This is a **retrospective computational result**, not a newly executed training run in this Pass. Frozen predictions were independently rescored and split identity independently reconstructed using actual source observations.

## Publication evidence

New genuine observed-data two-panel vector graphic:
`code/section_6_engineering_concrete_mix_generation/figures/concrete_verified_heldout.svg`

Full source:
`code/section_6_engineering_concrete_mix_generation/plot_verified_heldout.mjs`

Panel A plots all 229 genuine test observed strengths versus model predictions with identity line; Panel B shows 229 original residuals with zero-error reference. All numeric points are sourced directly from committed real-test records. Actual observed and residual ranges fit the figure limits. The generated SVG is source-linked to the CSV and accompanying generator, not a schematic, bar-chart placeholder, or fabricated experiment. Existing `publication_figures.py` remains available for other reported comparisons. Current manuscript placement and caption matching remain production HOLD.

## R1–R10 scoped decision

| Criterion | Result | Supporting evidence |
|---|---|---|
| R1 | PASS | Original Yeh real measurements with UCI provenance and immutable CSV identity |
| R2 | PASS | Exact independently recreated 645/156/229 grouped partition, 0 test recipe leakage |
| R3 | PASS | Already executed real-data five-neighbor model and empirical conditional proposal algorithm with code and frozen outputs |
| R4 | PASS | 229 original observed versus predicted strength pairs; 200 stored numerical candidates |
| R5 | PASS | Independent training-mean test baseline and validation-selected alternatives |
| R6 | PASS | Original-data held-out RMSE/MAE, zero-target-hit negative finding, exploratory screening limitations |
| R7 | PASS (offline scope) | Measured withheld concrete specimens evaluate strength prediction; does **not** validate physical properties of novel mixtures |
| R8 | PASS (scientific figure scope) | Original-number two-panel measured-data SVG plus generator and plot range audit; current manuscript integration separate |
| R9 | PASS | Complete original model script, original CSV, outputs, plotting scripts and independent audit source stored in GitHub |
| R10 | PASS | Strictly computational demonstration; no new physical strength or safe design claims |

**Level A certificate scope:** demonstrated authentic-data *retrospective* prediction and computational sampling/screening only. Does not certify empirical strength or durability of generated mixtures; structural engineering viability; prospective success; a full autonomous physical concrete mix-design system; or original parent application title if it promises these properties.

## Accounting and remaining gates

The scoped certification register now contains **5 explicitly scoped scientific certificates**: four Level A, one Level B. This is **not** five of fourteen fully certified parent chapter studies; no such parent-study claim is made. The complete current manuscript remains **PRODUCTION_HOLD**.

For the broad concrete physical-material application, the external laboratory must obtain new signed original specimen observations before any physical certification. For driving, obtain genuine BDD100K multi-scene experiment outputs and held-out downstream tests. For manuscript production, inspect the actual latest LaTeX/PDF and verify exact figure mapping.

## Audit artifacts

- `qa/audit_concrete_retrospective.py` independent original-row and summary checker.
- `qa/validate_scoped_ch6_certifications.py` register plus concrete and Brownlee record gate.
- `qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv`.
- Data-derived figure `code/section_6_engineering_concrete_mix_generation/figures/concrete_verified_heldout.svg`.
