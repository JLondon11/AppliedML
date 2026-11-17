# Chapter 6 — Pass 64: Master QA scientific–production reconciliation and exact LaTeX figure bindings

**Date:** 2026-11-16  
**Source policy:** repository-root `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md` (R1–R10, proportional Levels A/B/C). Authentic source measurements, actual executed task-specific algorithms, saved real results, complete committed code, accurate captions and source-linked figures remain compulsory. Scientific certificates and final production approval are **distinct**.

## Verified source state

The Chapter 6 GitHub root (`chapters/generative_ai_part_i/`) continues to lack a current canonical `manuscript/` TeX/PDF directory. The only directly accessible user-supplied handoff `Proceed Chapter 6.txt` is an earlier text record reporting Figure 6.20 on printed page 67 of a 109-page Pass 37 chapter with problems on page 101; its linked PDF/TeX/ZIP are **not present** in the active workspace. Therefore that source supports the historical reported placement only, not direct inspection of today's PDF layout or the scientific source/image/caption relationship.

The detailed current register records **12** explicitly restricted scientific experiments (11 Level A, 1 Level B), all `PRODUCTION_HOLD`. These do **not** certify the fourteen parent application/case-study titles. The older root `CERTIFICATION_REGISTER.csv` separately contains **unmapped, provisional** parent records and an older VAE/PCA partial evidence status; it must not be mistaken for a fresh complete parent-application certification, nor should it silently supersede the newer *scoped* certificate register. Both registers need reconciliation to the current exact manuscript when available.

## Concrete production defect fixed

The prior production gate could trust a `production_inventory.json` caption and SHA-256 without proving that the actual LaTeX contains the corresponding `figure` environment, label, image path or caption. This could permit a **different manuscript figure to be signed off** merely because a correctly hashed asset existed elsewhere in the repository. This is especially material for the disputed industrial-defect Figure 6.20 and its unrelated legacy handwritten-digit script filename.

Added `qa/verify_tex_figure_bindings.py`, with strict source-linked inspection:

- Load the actual main `manuscript_tex` and follow local `\\input`/`\\include` source references, rejecting absent and unsafe paths.
- Ignore comment-only figure code and require exactly one inspected label, caption and graphic per recognized figure environment.
- Match **all** actual figure environments 1:1 against all inventory figures, reject duplicate labels or unlisted figures, and require each exact caption and graphic reference to agree with the current LaTeX.
- Resolve the **actual** LaTeX graphic to the **identical on-disk image asset** referenced by the inventory, not merely an item with the same figure number or reported caption.
- Fail closed when the real inventory or TeX is missing, or when custom LaTeX cannot be unambiguously parsed. This tool is a static source check, not LaTeX compilation, physical scientific validation, or PDF visual review.
- Integrate this check as mandatory `latex_figure_bindings` in `qa/check_ch6_release_gate.py`, so no chapter production structural approval is possible without the exact current binding.
- Add required real-source fields `manuscript_tex`, figure-specific `tex_file`, `tex_label`, and `tex_includegraphics` to `quality_assurance/PRODUCTION_INVENTORY_SCHEMA.md`.

## Actual executed regression checks

A local integration package was assembled from the previously executed original GitHub Pass 59 release/production files plus the **exact new GitHub-source-bound** TeX checker and test suite. Current Git blob SHA-1 identities were checked against the GitHub fetched versions:

| Code/test source | Git blob SHA-1 |
|---|---|
| `qa/check_ch6_release_gate.py` | `764dbf969bde9d78e0377ff5f37f5e5a2be84cca` |
| `qa/test_ch6_release_gate.py` | `e40ab72f9901c5ceaeb8a0a4c6412f0bf6e1e958` |
| `qa/verify_tex_figure_bindings.py` | `1b53fc4cfc253da9ee3c2502e2d3b09ccdca0d04` |
| `qa/test_verify_tex_figure_bindings.py` | `2d245ed5a650abcb86721afa0f303ce55e7c313f` |

Executed `python -m unittest discover -s qa -p 'test_*.py' -v` on all archived production inventory tests, integrated release-gate tests and the new TeX source/asset-binding cases: **34 tests, 34 passes, 0 failures**. Included negative cases for mismatched captions, actual image path, unrelated/detached asset, missing images, duplicate labels, omitted figures, commented-out source, absent inventory and unsafe includes.

The downloadable `Chapter6_Pass64_Production_Figure_Binding_QA.zip` contains exact tested Python files, execution log and a hash manifest. Fixture files are artificial unit-test inputs only and are never mistaken for scientific measurements or book graphics.

## Updated workflow

`.github/workflows/chapter6_scoped_level_a_certification.yml` now runs the new binding regression suite and reacts to changes to the checker/test source. The GitHub workflow's completed execution has **not** been independently confirmed. The release checker itself blocks when the current inventory/TeX is missing.

## Remaining Master QA release requirements — not waived

1. **Current exact compiled LaTeX and PDF:** obtain and inspect the complete actual chapter source/assets, not historical link names; render every page and visually check Figure 6.20, float position, captions, readable scientific figures, unresolved references, blank-space defects, final problem section, tables, grammar, section depth and production notes.
2. **Current full inventory:** reconcile all genuine figures, 14 parent applications, 17 historical tables and 11 historical numbered listings (or document a newly audited compiled count). Every empirical parent must have actual results/figures and full scope-matched R1–R10.
3. **Driving:** genuine multi-scene BDD100K (or comparably task-matched real driving data), generated outputs and independent held-out full-scene downstream perception evaluation are necessary for a driving-synthesis/perception-benefit claim. A one-scene stereo patch demonstration is not enough.
4. **Concrete:** the measured Yeh source validates retrospective strength predictions only; no independent cast/test results for new designed mixtures have been presented. Retain computational conclusions only until true physical testing.
5. **Figure 6.20:** the current `quality_assurance/chapter6_evidence_manifest.json` identifies a **provisional, unverified** handwritten-digit candidate for an industrial-defect figure. Its number/name collision must not be interpreted as scientific provenance. The new gate will prevent unverified actual LaTeX/asset mapping from being marked production approved.

**Decision:** Additional production QA software and its tests **PASS**, 12 existing narrow scientific certificates retained, **zero** new broad parent application scientific certifications, full Chapter 6 scientific/production release **HOLD** under Master QA. Do not claim full book compliance merely because unit-test fixtures passed.
