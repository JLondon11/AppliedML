# Chapter 6 — Pass 64 independent original-data figure reproduction and production QA

**Review date:** 2026-11-16. **Current scientific status:** 12 carefully scoped certificates retained (11 A, 1 B); no new claim of 14 fully certified original broad applications. **Production status:** Figure staging passed, complete chapter publication **HOLD**, manuscript source/PDF inaccessible.

## Discovered real delivery reproducibility defect

The Pass 63 downloadable ZIP `Chapter6_Pass63_Seven_Print_Width_Scientific_Figures.zip` contained `build_compact_real_image_plates.py`, the source PDF figures and their provenance, but **did not include** the Python source module `audit_true_tissue_vae_vs_nearest.py` imported by the tissue checkpoint figure builder. Re-executing the distributed Pass 63 script in an isolated extraction raised **`ModuleNotFoundError: No module named 'audit_true_tissue_vae_vs_nearest'`**. The figures were authentic and had already passed original figure QA, but their purported delivery package was not self-contained.

## Performed remediation

Recovered the original exact model definition and real-photo patch extraction from the *previously delivered actual Pass 62* original package, and included it in a new ZIP beside the original real-photo-trained VAE checkpoint (SHA256 `204d5bd26a3361584285f108c7e74b9bdad2e5ee9d52310c7a5204f12d08bbbe`). Preserved original authentic source microscope pixel SHA256 `c5b3ef509a92f16d4c29be8cf0300fe75d53e13a3ce650159db932caea8dcc1b`. Reused the exact unchanged original input digit scans and RDKit/NCI original chemical structure source. Models were **not freshly refitted** or substituted with imagined results.

The seven figures were all **successfully regenerated in a fresh output directory** using the delivered original source files plus the included actual previously trained checkpoint. Source and checkpoint identities matched their original experimental records.

## Strict independent rendered-image parity

For all seven original and freshly regenerated PDFs, the audit examined the true one-page PyMuPDF rasterized RGB image (72 dpi, identical renderer and environment) and compared *every channel in the full image*, rather than comparing JPEG thumbnails or report summaries. **Result: seven regenerated PDFs had ZERO pixel-level RGB channel differences from the exact delivered Pass 63 PDFs.** This includes every original observed digit, tissue source region, NCI chemical structure, actual VAE output image, and the two source-faithful conceptual architecture diagrams. There were no silently changed model predictions.

| Replayed exact scientific asset | Original-data/model raster panels | Minimum actual font | Result |
|---|---:|---:|---|
| `tissue_source_compact.pdf` | 7 | 8.2 pt | exact pixel parity |
| `digits_source_compact.pdf` | 20 | 8.2 pt | exact pixel parity |
| `nci_source_compact.pdf` | 6 | 8.2 pt | exact pixel parity |
| `tissue_vae_output_compact_1.pdf` | 16 | 8.2 pt | exact pixel parity |
| `tissue_vae_output_compact_2.pdf` | 16 | 8.2 pt | exact pixel parity |
| `smiles_vae_architecture_compact.pdf` | 0 (native vector) | 8.2 pt | exact pixel parity |
| `tissue_vae_architecture_compact.pdf` | 0 (native vector) | 8.2 pt | exact pixel parity |

The actual PDF widths are all 468 pt (the **provisionally assumed** 6.5-inch Chapter 6 textbook text measure); none has text outside page bounds, none has less than 7.5 pt selectable text, and all have the scientifically appropriate original/model image counts.

## New complete user-facing deliverable

`Chapter6_Pass64_Reproducible_Figure_Portfolio.zip` contains the seven PDF/PNG/SVG source/model output graphics, exact experimental source figure generators, the formerly missing `audit_true_tissue_vae_vs_nearest.py` model module, the actual pretrained-by-this-experiment VAE checkpoint, `verify_pass64_replay.py` full replay test, actual `PASS64_VERIFIED_REPLAY.json`, an exact SHA256 file inventory and a README with instructions. The completed ZIP has 32 files and passed a physical `zipfile.testzip()` archive integrity test.

An independently implemented reusable `qa/verify_pass64_figure_replay.py` was committed to GitHub, with the same source authenticity, self-contained dependency, checkpoint, physical print-width, image-count, off-page text and exact reconstructed-raster parity checks. **Its local equivalent was executed and passed; remote GitHub Actions runs are not asserted successful.** The binary ZIP and checkpoint are delivered to the user, not falsely claimed committed into the GitHub text-file tree.

**This correction prevents the false inference that a good-looking scientific figure is necessarily independently reproducible from the delivered code alone.**

## Remaining broad scientific and production gates

The latest full Chapter 6 TeX/PDF is still absent from the accessible repository and this current workspace. Checking the GitHub repository release inventory returned no release assets. The earlier `Proceed Chapter 6.txt` handoff names a Pass 37 Figure 6.20 PDF and source paths, but those exact referenced files are not available. The handoff is historical placement evidence, **not an available final compiled document that can now be inspected**. The current Figure 6.20 industrial-defect to source/asset mapping remains unverified, and no new exact page/number is invented.

Similarly, the original broad generative driving and newly generated concrete physical performance claims cannot be certified without their task-matched genuine multiscene/physical experiments. The figure-portfolio reproduction does **not** certify those applications or produce additional scientific certificates.

**Pass64 decision:** original scientific figure portfolio delivery **repaired and fully replay verified**; seven print-width staged science graphics supported by authentic data and executed model artifacts; book-wide/chapter-wide scientific-parent and final manuscript production status **HOLD**.
