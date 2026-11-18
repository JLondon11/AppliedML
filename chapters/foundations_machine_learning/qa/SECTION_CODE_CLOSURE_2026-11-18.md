# Chapter 1 Section-Organized Application/Case-Study Code Closure

Date: 2026-11-18

## Scope

The 21 certified Chapter 1 applications and case studies were audited against the manuscript provenance matrix for:
- section number
- real dataset
- model family
- metric/evaluation role
- generated result figure role
- executable code presence

## Repository organization

Canonical code root:
`chapters/foundations_machine_learning/code/sections/`

The following section directories are present:
`1_8_1` through `1_8_5`, `1_9_1` through `1_9_6`, `1_21_1` through `1_21_9`, and `1_22_5`.

Each directory contains a runnable `run.py`.

## Execution verification

All 21 scripts were syntax-checked and executed locally against authentic scikit-learn or statsmodels datasets. Each execution produced:
- numerical metrics
- a generated result figure
- zero runtime failures after remediation

Method-level mismatches found during audit were corrected before publication, including:
- top-quartile credit-risk target definition for 1.8.1
- validation-MSE transformer diagnostics for 1.8.3 and 1.9.3
- second-difference hydrologic regularization for 1.8.4 and 1.9.4
- Adam learning-rate sensitivity for 1.9.2
- online logged-choice learning for 1.9.5
- native-unit cross-domain diagnostics for 1.9.6
- quadratic detrending for industrial condition monitoring in 1.21.8

## Final status

**21/21 section-organized application/case-study code units: PASS.**
**21/21 executable locally: PASS.**
**Repository section organization: PASS.**
**Dataset/model/metric alignment: PASS.**
