# Chapter 6 — Pass 60: exact scoped certificate CSV repair and source-image scientific QA

**Date:** 2026-11-14. **Outcome:** corrected a *real* twelve-unit scientific register schema defect, introduced a fail-closed validator and regression tests, and completed four image-first figure-asset source audits. This report does not claim final published chapter approval.

## Discovered malformed scientific register

The current `qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv` had a proper **17-column header**, yet **D12** (the real original handwritten-digit conditioned latent diffusion demonstration) had **18 values**, including an accidental **eleventh `PASS`** between ten R1–R10 values and the science/production statuses. Such a row causes Python `csv.DictReader` to silently put excess information into its unnamed column and shifts science/production fields, potentially invalidating certificate accounting without an obvious parsing error.

The anomalous row was directly verified in the live GitHub source, and the extraneous `PASS` field was deleted **without changing the underlying ten recorded R1–R10 PASS judgements or the scientific evidence**. D12 now has the correct *ten* requirements and separate `SCIENTIFIC_CERTIFIED` and `PRODUCTION_HOLD` columns.

After the update, an independent fresh GitHub source read-back with a quote-aware CSV parser verified:
- **12 distinct consecutive scientific-unit IDs D01–D12**;
- **17 columns on all 12 rows plus header**;
- **10 PASS criteria per row**, correctly positioned, none missing or extra;
- **12 recorded scoped SCIENTIFIC_CERTIFIED** and **12 PRODUCTION_HOLD**;
- **11 Level A and 1 Level B**, preserving the legitimate comparative NEU study as B; and
- explicit nonblank source-scope limitations.

This is a **certificate-register data-integrity fix**, not new model scientific measurements and **not certification of twelve broader parent applications**.

## Permanent fail-closed remediation

New `qa/validate_scoped_csv_schema.py` uses strict `csv.reader` with exact field-count checks, correct header, contiguous IDs, title/claim limitation presence, true R1–R10 positioning and scientific/production statuses. It rejects false Chapter 6 production approval while the latest compiled chapter remains absent.

New `qa/test_scoped_csv_schema.py` defines **six negative/positive fixture scenarios**: a valid 12-unit structural-only fixture, extra eleventh PASS, R requirement PARTIAL, deletion of a scope row, gapped/duplicated ID, and falsely marked PRODUCTION_APPROVED. These fixtures are **not actual model experimental results**. They are committed and wired into `.github/workflows/chapter6_scoped_level_a_certification.yml`. **Remote GitHub Actions execution has not yet been independently confirmed**, so the source code is delivered as a QA enforcement improvement, not a claimed completed CI run.

## Four actual image-first scientific figures, not just plots

The staging ledger `quality_assurance/pass59_image_first_figure_staging.json` lists **four** PDF/PNG/SVG deliverables, all produced in the local runtime, rendered to bitmap at print-preview scale and visually checked:

1. **Original microscopy photo and source-train/validation/test patches** (input source imagery, not reconstructions).
2. **Original scanned handwritten-digit gallery** (two true scans per class, not generated outputs).
3. **Authentic original tissue VAE model-output comparison**, newly executed trained checkpoint: original held-out tissue → *actual* VAE reconstruction → spatial absolute RGB error → VAE-generated prior sample. Original single tissue 589/93/217 train/validation/test patches, **217 actual test outputs**, mean PSNR **20.1552659471 dB**, SSIM **0.2926715016**, pixel MSE **0.0104811601**. This new local execution uses an earlier documented seed and source architecture but has slightly different metrics from a previously archived run; do not falsely claim identical replay. The gallery objectively shows blurred reconstructions and uninformative prior-generated tissue.
4. **Original chemical NCI molecular structure plate**, featuring twelve actual RDKit chemical-source molecules selected from authentic 360/120/120 original train/validation/test molecular records, with true RDKit atom/bond depiction and **calculated** empirical molecular formulas. These are *original input molecules*, not VAE-generated structures and not measured new drug activity.

All four generated figure PDFs were independently hashed. The exact local delivered PDF SHA256 values and original-source hashes are recorded in the staging ledger. The local `qa/audit_pass59_image_first_delivery.py`-equivalent checker was **executed against all four actual PDFs** plus authentic source datasets and the 217-patch model data record, yielding **PASS_STAGED_SCIENTIFIC_FIGURE_ASSET_INTEGRITY**. This is a *local staged-asset integrity* result. The new committed source has corresponding QA logic and image re-execution Actions workflow.

Source paths, all committed:
- `figure_generation/original_source_image_plates/make_authentic_source_image_plates.py`
- `code/section_6_medical_imaging_histology_vae/plot_tissue_original_model_outputs.py`
- `code/section_6_molecular_smiles_vae/make_original_nci_structure_plate.py`
- `qa/audit_pass59_image_first_delivery.py`
- `.github/workflows/chapter6_image_first_authentic_figures.yml`

No generated figure is described as incorporated in a chapter manuscript without actual TeX/PDF inspection. User evidence ZIP retains figure PDF/PNG/SVG, measurements and available original checkpoint, complete displayed-code source and numeric provenance.

## Latest full production gate, unchanged HOLD

- The user-provided `Proceed Chapter 6.txt` states a **historical** Figure 6.20 placement correction on printed page 67 of a 109-page Pass37 manuscript. This note is not the actual latest production PDF/TeX.
- The current connected repository has **no complete compiled current Chapter 6 PDF or TeX source**. Its figure manifest still associates industrial-defect Figure 6.20 with an **unverified handwritten-digit** `figure_20.png` script candidate, and `quality_assurance/production_inventory.json` is absent. Therefore full 48-figure, 17-table, 11-listing, 14-parent production traceability cannot honestly be approved.
- Existing multi-scene BDD100K downstream synthetic-driving perception and independently measured strength of new generated concrete mixes remain scientific HOLD **for applications making those broad claims**. The older narrow experiments are correctly distinguished and remain separately scoped.

**Pass60 verdict:** scoped scientific register **correctly structured: 12/12 valid records**; actual staged image-first figure integrity **4/4 verified locally**; new genuine tissue model run **executed**; final full Chapter 6 scientific-parent approval **HOLD**; final manuscript production approval **HOLD** until the current manuscript and all 48 actual production assets are available and fully inspected.
