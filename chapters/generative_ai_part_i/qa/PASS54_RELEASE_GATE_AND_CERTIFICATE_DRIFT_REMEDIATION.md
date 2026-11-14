# Chapter 6 — Pass 54: certification-drift repair and fail-closed production gate

## Confirmed current repository state
On inspection of `main`, the scoped scientific register lists six uniquely identified experiments: CH6-D01 through CH6-D06. Five are **Level A**, and CH6-D04 is **Level B**. All ten recorded R requirements are PASS for each *limited stated experimental scope*, with `scientific_status=SCIENTIFIC_CERTIFIED` and `production_status=PRODUCTION_HOLD`. This is **not** six of the fourteen full parent chapter applications.

The six scopes, with references to source records, are:
1. D01 Brownlee retrospective static process emulator;
2. D02 genuine handwritten-digit VAE/PCA reconstruction;
3. D03 observed Mauna Loa CO2 sequence forecasting and calibration failure;
4. D04 authentic NEU steel-defect GAN augmentation (negative effect in tested protocol);
5. D05 original Yeh concrete records, retrospective strength prediction and *unvalidated numerical formula proposals*;
6. D06 original Middlebury stereo images, within-one-scene conditional view synthesis.

The current `RELEASE_MANIFEST.md` previously referenced only the four Pass 50 certificates and the previous `test_scoped_certification_validator.py` asserted four. Both are now updated to six; the register's actual scientific claim scopes remain unchanged.

## Actual Figure 6.20 / manuscript inspection gate
GitHub's Chapter 6 root currently contains directories `code/`, `figure_generation/`, `figures/`, `production_pass14/`, `qa/`, and `quality_assurance/`, but **not** a `manuscript/` directory with current exact .tex/.pdf artifacts. A prior Pass 45 release reportedly delivered a compiled 109-page chapter outside this repository; that statement is historical and is not an inspection of the current committed manuscript.

The source `quality_assurance/chapter6_evidence_manifest.json` explicitly marks printed Figure 6.20's link to a candidate handwritten-digit `figure_20.png` as unverified and domain-mismatched against an industrial-defect figure. A colliding filename does **not** prove the wrong image was inserted, but it prohibits accepting the candidate as the figure's scientific source.

## New executable gate
`qa/check_ch6_release_gate.py` is a fail-closed standard-library gate checking:
- exactly six authentic scoped-register IDs, correct level/status assignments and R1–R10 recorded values;
- Figure 6.20 matched manuscript/evidence domain, inspected mapping, caption, committed source, output file and expected SHA-256; and
- existence of the current release manuscript's TeX and PDF files.

It returns nonzero if any condition is missing. A structural pass returns `STRUCTURAL_GATE_PASS_REVIEW_REQUIRED`, **never** automatic scientific or production certification.

`qa/test_ch6_release_gate.py` adds five deterministic fixture-level regression cases: missing evidence, scoped results without manuscript, mismatched Figure 6.20 domain, missing caption, and a passing *structural-only* fixture.

### Validation performed in this pass
- Inspected the current GitHub root directory listing; confirmed manuscript directory absent.
- Reopened the current CSV register and verified six unique IDs, five A/one B, and all recorded R1–R10 PASS within stated scopes.
- Inspected the actual Figure 6.20 manifest with `manuscript_mapping_verified=false` and a domain mismatch.
- Checked newly committed source/test code by source inspection. **No independent Python execution of these two newly committed QA files was performed in this pass**; do not report the fixture tests as passing until run.

## Next scientific priorities
1. Add exact current manuscript .tex/.pdf and all accepted figure assets in the designated release source or an immutable artifact reference; review Figure 6.20's actual image/caption/source hash.
2. Execute genuine BDD100K multi-scene generation plus independent held-out full-scene downstream perception evaluation. Crop-classification software fixtures cannot substitute.
3. Obtain independently signed actual physical specimen results before making a claim about properties of newly generated concrete mixtures.
4. Review all fourteen *parent* application titles against their exact evidence and figures, without promoting the six narrow certificates.

**Overall Chapter 6 production release: HOLD.**
