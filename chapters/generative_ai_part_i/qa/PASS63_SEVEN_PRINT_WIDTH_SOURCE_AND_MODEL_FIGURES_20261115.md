# Chapter 6 — Pass 63: seven genuine-data, source-faithful and actual-model-output print-width plates

**Date:** 2026-11-15. **Scientific outcome:** no additional unit artificially certified; the existing twelve appropriately scoped experimental certificates retain their scope. **Production outcome:** seven redesigned figures passed a real 468 pt (6.5-inch) source-PDF font and clipping preflight. **Final chapter production still HOLD** because the latest full manuscript PDF/LaTeX is absent.

## What Pass 63 remediates

Pass 62 found that all seven original wide staged figures would have tiny typography if physically inserted at an ordinary 6.5-inch text width (previous smallest printed labels approximately 3.4–4.6 points). Merely reading the wide PDF at full zoom was inadequate. Pass 63 **regenerated the scientific source/modeled content in narrower layouts**, not by shrinking the old PDF and pretending it was legible.

All seven Pass 63 assets are **468 pt wide** with a real measured **smallest vector-font size 8.2 pt**, above the 7.5 pt preliminary type floor. The actual PDFs were rendered and visually inspected for the source plate and representative model-output, chemistry, and architecture plates. A second strict PyMuPDF pass examined all pages for off-page text, a defect initially found in the three input-source figure titles and corrected before final delivery. Final **seven PDFs: zero off-page text spans, correct original-image/model-output counts, one page each, and 7/7 staged print gates passing**.

| Scientific figure | Type and original-source meaning | Actual observed/model image panels | Stage width | Smallest text |
|---|---|---:|---:|---:|
| `tissue_source_compact.pdf` | Authentic tissue photo with source-row training/validation/test regions and six real microscope crops | 7 | 6.5 in | 8.2 pt |
| `digits_source_compact.pdf` | Twenty genuinely scanned 8×8 digit observations (two per label 0–9) | 20 | 6.5 in | 8.2 pt |
| `nci_source_compact.pdf` | Six genuine RDKit NCI chemical source graphs from correct original molecular train/validation/test groups | 6 | 6.5 in | 8.2 pt |
| `tissue_vae_output_compact_1.pdf` | Four authentic held-out tissue patches versus actual checkpoint neural reconstructions, spatial errors and prior generations | 16 | 6.5 in | 8.2 pt |
| `tissue_vae_output_compact_2.pdf` | Four additional genuine held-out/model output examples, no synthetic observation claim | 16 | 6.5 in | 8.2 pt |
| `smiles_vae_architecture_compact.pdf` | Vector pedagogical representation of actual character embedding/BiGRU/latent-32/autoregressive GRU implementation | Vector, not a raster input | 6.5 in | 8.2 pt |
| `tissue_vae_architecture_compact.pdf` | Vector pedagogical representation of the actual three-convolution/latent-24/transposed-convolution VAE | Vector, not a raster input | 6.5 in | 8.2 pt |

**Scientific claim boundaries:** the tissue images originate from **one** authentic immunohistochemistry source photograph with correlated derived patches. The digit scans are original inputs, not generated outputs; NCI chemical graphs represent original source structures, not discovered/synthesized drugs. Prior tissue samples are produced by the previously executed genuine source-trained checkpoint, not patient photographs. Architecture drawings are explanatory, **not executed numerical experiments**. The actual model-output gallery must not be recharacterized as independent clinical testing.

## Complete code and results delivered

- `figure_generation/print_width_original_image_portfolio.py`: authentic scikit-image microscope photo, actual sklearn handwritten scans, original RDKit NCI molecule source checks, checkpoint-driven real tissue reconstructions and generated prior outputs.
- `figure_generation/print_width_architecture_plates.py`: executable two-model source-code-specific narrow scientific schematics.
- `qa/validate_pass63_print_figures.py`: exact seven-name and seven-page fail-closed audit, original image-panel counts, selectable-text presence, 468 pt page width, physical minimum font at least 7.5 pt and no text bounds outside the PDF page.
- `.github/workflows/chapter6_image_first_authentic_figures.yml`: independent actual original-photo VAE rerun with all genuine source dependencies, generation of all seven compact figures and automatic physical-font/text-bounds verification.
- The downloadable user evidence package contains seven actual vector PDF/PNG/SVG triplets, original-source and original-checkpoint SHA256 provenance, and a **local executed print preflight report** with exact delivered PDF hashes. The figure binaries are **not** claimed committed to GitHub.
- Independent local runtime verification of all seven finished PDFs returned **7 PASS / 0 HOLD**, each page exactly **468 pt** wide, with **zero off-page text spans** and minimum actual font **8.2 pt**.

## Publication integration controls

Use figure captions to distinguish the different kinds of evidence: original inputs/partition, actual trained-VAE outputs, and conceptual model architectures. Do not replace the chapter's case-study empirical result figures with conceptual schematics. Reserve Figure 6.20 until the correct industrial-defect model-output asset is identified from the actual current manuscript; a handwritten-digit file with a matching provisional number is **not** adequate provenance.

**The measured 6.5-inch figure width is an explicit assumed layout constraint** from the previous QA, not verified against the unavailable latest typeset chapter. Figure numbering, exact insertion points, page breaks, table content, and final chapter image quality cannot be certified without actual manuscript source and compiled PDF. Additional page/figure space may still need editorial selection: seven staged figures are a diverse candidate library, not an instruction to insert all of them redundantly.

The 14 full named parent applications and case studies remain separately scientifically incomplete where their broad stated tasks require authentic missing experiments (for example BDD100K multi-scene driving downstream perception and genuinely measured physical strength of newly designed concrete mixes). **No broader parent application is silently certified by these production graphics.**

**Pass63 final decision:** seven authentic and technically informative figure candidates are print-width-ready by strict staged preflight; genuine scientific certification scope remains unchanged; chapter-wide scientific and manuscript production release remain HOLD.
