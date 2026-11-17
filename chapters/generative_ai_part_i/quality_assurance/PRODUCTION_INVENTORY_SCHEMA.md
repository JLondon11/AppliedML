# Chapter 6 — Exact production evidence inventory specification

This describes the required **actual** `production_inventory.json` assembled from the latest compiled LaTeX/PDF and matching assets. It is not a substitute for inspecting the chapter and contains no invented figure mappings.

## Source of truth and last audited baseline

The last archived compiled Pass 44/45 chapter was 109 pages with **48 figures, 17 tables, 11 numbered code listings and 14 named applications/case studies**. These are historical counts. Until an independently rebuilt revision with a newly checked count inventory is supplied, `qa/verify_ch6_production_inventory.py` requires complete coverage of this baseline. Do not alter the code's counts merely to make tests pass. If editorial remediation legitimately removes, adds, or renumbers items, update the audited baseline **with the newly compiled manuscript, its asset manifest and the review report**, then reconcile all references.

## Record schema (real values only)

At `quality_assurance/production_inventory.json`:

```json
{
  "chapter": 6,
  "compiled_source_reviewed": false,
  "manuscript_tex": "chapters/generative_ai_part_i/manuscript/<actual-main-file>.tex",
  "manuscript_pdf": "chapters/generative_ai_part_i/manuscript/<actual-current-file>.pdf",
  "manuscript_sha256": "<actual-64-character-SHA256>",
  "figures": [
    {
      "figure_id": "6.20",
      "asset": "<actual-repository-relative-image-path>",
      "tex_file": "<actual-repository-relative-TeX-file-containing-figure>",
      "tex_label": "<exact-label-from-LaTeX-figure-environment>",
      "tex_includegraphics": "<exact-image-path-inside-includegraphics>",
      "sha256": "<actual-image-SHA256>",
      "caption": "<caption-copied-from-current-LaTeX>",
      "scientific_reviewed": false,
      "kind": "experimental",
      "dataset_provenance_verified": false,
      "generator": "<actual-committed-generator-path>",
      "observed_data_or_results": "<actual-saved-real-data/output-path>"
    }
  ],
  "parent_applications": [
    {
      "unit_id": "<actual-application-section-ID>",
      "exact_manuscript_title": "<exact-current-section-title>",
      "scope_reviewed": false,
      "figure_ids": ["<real-figure-ID>"],
      "scientific_status": "SCIENTIFIC_HOLD"
    }
  ],
  "tables": [
    {
      "table_id": "<real-table-ID>",
      "substantive_reviewed": false,
      "source_or_derivation": "<true-data-or-scientific-derivation-reference>"
    }
  ],
  "code_listings": [
    {
      "listing_id": "<real-listing-ID>",
      "complete_code": "<actual-committed-script-path>",
      "code_reviewed": false
    }
  ]
}
```

**These are schema examples only**. Do not commit them as genuine records, and do not manufacture values. For a nonempirical conceptual figure set `kind: "conceptual"` and supply a scientific/visual review with a correct caption and actual image hash. Empirical parent case studies must each cite at least one traced original-results figure and have scientifically reviewed scope. The 14 parent applications remain individually distinct from the current 12 narrower scientific certificates.

The production release checker also requires the actual latest manuscript TeX/PDF and independent Figure 6.20 industrial-defect figure source-to-caption verification. A legacy digit figure named `figure_20.png` is not evidence that it belongs in the industrial-defect section. Neither missing BDD100K full-scene perception experiments nor absent independent casts/tests of generated concrete may be concealed by a publication figure.

## Run

```bash
python chapters/generative_ai_part_i/qa/verify_ch6_production_inventory.py --repo-root .
python chapters/generative_ai_part_i/qa/check_ch6_release_gate.py --repo-root .
python -m unittest discover -s chapters/generative_ai_part_i/qa -p 'test_ch6*.py' -v
```

A passing *structural* inventory still requires manual review of manuscript content, visual layout, scientific conclusions and code correctness. It does not grant whole-book Master QA approval by itself.

## Direct LaTeX figure binding gate (Pass 64)

The new `qa/verify_tex_figure_bindings.py` is run by the Chapter 6 release gate **in addition to** the inventory/PDF hash checks. Starting from `manuscript_tex`, it follows local `\\input`/`\\include` source links, strips commented-out figure code, requires exactly one `\\label`, one `\\caption` and one `\\includegraphics` per supported figure environment, and compares these with **every** indexed figure. It also requires each actual graphic path to resolve to the **same on-disk asset** supplied in the inventory. Missing TeX, unsafe includes, duplicate labels, omitted inventory figures, caption differences, and unrelated images block release.

The checker intentionally fails closed on complicated/custom LaTeX it cannot statically disambiguate; resolve those cases by an independent, documented source-build audit rather than claiming a false PASS. A complete source-binding result remains `STRUCTURAL_TEX_BINDINGS_PASS_REVIEW_REQUIRED`: it does not prove scientific figure correctness, PDF pagination, float placement, complete production or any empirical R1–R10 requirement. Actual PDF visual inspection remains necessary.

Unit test: `python -m unittest discover -s chapters/generative_ai_part_i/qa -p 'test_verify_tex_figure_bindings.py' -v`.
