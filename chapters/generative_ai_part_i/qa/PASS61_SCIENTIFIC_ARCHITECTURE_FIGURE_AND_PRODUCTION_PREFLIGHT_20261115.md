# Chapter 6 — Pass 61: scientific architecture, authentic-model evidence and production preflight

**Review date: 2026-11-15.** **Result:** a scientific **architecture schematic** was generated from the actual character-level SMILES VAE code, visually inspected and added to the image-first staging ledger. The existing four authentic scientific image galleries were re-inspected. Their exact delivered original PDF SHA-256 hashes were checked again, and the 217 measured original VAE held-out tissue-patch records were independently reconciled with the corresponding experiment metadata.

## User-required figure diversity: scientific figures need not all be plots

The five staged scientific figures have the following genuinely different informative roles:

| Staging ID | Figure type | Correct scientific role | Empirical model output? |
|---|---|---|---|
| PASS59-SOURCE-TISSUE | Original tissue photograph, annotated spatial regions and actual crops | Original data, visual morphology, explain split/leakage safeguards | No — source observations |
| PASS59-SOURCE-DIGITS | Two original handwritten scans for each 0–9 digit | Reveal natural real scanned input diversity and resolution | No — source observations |
| PASS59-SOURCE-NCI | Twelve actual original NCI molecular chemical graph drawings | Authentic chemical source identities, split groups, molecular structures | No — RDKit-depicted original inputs |
| PASS59-PREDICTION-TISSUE | Held-out original tissue vs trained CVAE reconstructions, pixelwise absolute error, separately sampled priors | Visualize the actual executed neural model's successes and failures | **Yes — genuine executed model output** |
| PASS61-ARCH-NCI | **BiGRU molecular VAE architecture/source-code diagram** | Explain character embeddings, two-headed Gaussian encoder, reparameterization, autoregressive decoder and RDKit checks | No — scientific conceptual architecture |

These five images are **not five statistical plots**. Do not use an original input gallery or a conceptual diagram in place of empirical R4/R8 output evidence when a case-study evaluation requires actual model predictions. Complement these informative visuals with genuine measured error distributions, experimental photographs and source-linked metrics where appropriate.

## Original NCI model architecture cross-check

Original complete source: `code/section_6_molecular_smiles_vae/run.py`; new diagram generator: `code/section_6_molecular_smiles_vae/plot_nci_smiles_vae_code_architecture.py`.

The original executed code specifies a 32-dimensional character embedding, **bidirectional GRU encoder with hidden width 128 each direction**, two independent Gaussian parameter heads `mu` and `logvar` from the concatenated 256-dimensional hidden state, **32-dimensional latent variable** and stochastic Gaussian reparameterization. It uses an autoregressive GRU decoder of hidden width 128, with a 32-dimensional latent vector concatenated with each 32-dimensional embedded token, the decoder hidden state initialized from a learned `z` projection, and a token-softmax output. Training uses teacher-forced next-character cross-entropy on non-padding tokens plus a scheduled `β × KL / 32` term, and selects the best validation cross-entropy checkpoint.

The original 600 valid uniquely canonicalized RDKit source records are divided into 360 training, 120 validation and 120 held-out test molecules. Two actual evaluation modes are distinguished: greedy held-out SMILES reconstruction and 1,024 stochastic prior-decoded strings, both checked using RDKit validity/canonical equivalence and (where appropriate) Morgan fingerprint similarities. The figure correctly states that a valid generated SMILES is **not** a measured new chemical compound, bioactivity, safety profile or physical synthesis. The original model's poor validity remains a measured negative result.

The new architecture schematic was actually authored and rendered as **PDF/SVG/PNG**, inspected as a PDF-rendered print-preview image, and corrected once to remove crossing arrows and text overlap before acceptance. The final staged PDF SHA256 is `c81550c20fda626fce8848523b8946c9852dd164bbbd0f50d81b49bec1975662`. It contains **vector-rendered shapes and selectable text**, not a screenshot of invented data.

## Executed five-asset PDF integrity and production-scale checks

The actual previously generated Pass60 ZIP was opened in the current runtime, including the four genuine original input and model result figures and their provenance records. The original archive-integrity checker was rerun and passed for **all four original PDF SHA256 values**, both original source image identities, 600 original NCI molecules and the newly re-examined 217 held-out actual model test patches. **The Pass60 bundled audit and new Pass61 source-equivalent local checks were executed; the new remote GitHub Actions workflow remains unverified.**

`qa/audit_pass61_scientific_figure_portfolio.py` is a new committed preflight for the five staged PDFs. It checks each original PDF's exact expected hash, one-page format, actual embedded-image counts, minimum readable text size, fully in-bounds text, the presence of scientific architecture terms and the different figure types. A local independent PyMuPDF equivalent was executed against all five delivered PDFs: **five PDFs passed, zero clipped/off-page text spans, no fonts below 6 pt, and all expected original-source/model embedded image counts matched**. The new architecture PDF has zero raster images because it is a native vector figure.

New independent original-model numerical reconciliation from the Pass60 archive:

- **217 original tissue test-patch rows**, with 217 unique `(source_row,source_col)` positions entirely in original source photograph rows **384–511**.
- Per-patch means recalculated directly from the actual CSV: PSNR **20.1552659471 dB**, SSIM **0.2926715070**, pixel MSE **0.0104811602**.
- These reconcile within numerical precision with the original executed-model report (PSNR **20.1552659471 dB**, SSIM **0.2926715016**, MSE **0.0104811601**). Do not count 217 patches from the same source as independent patient measurements.

## Production staging, not false release

The exact candidate figure names, source paths, claim restrictions, original stage hashes, and inspection statuses are in `quality_assurance/pass59_image_first_figure_staging.json`. Every item has `publication_figure_number:null` and `manuscript_placement_verified:false`. This is intentional: **the connected GitHub tree does not currently contain the latest compiled Chapter 6 PDF/TeX**, so actual printed figure numbers/pages/captions, model-source-to-manuscript matching and figure placement cannot be assigned truthfully.

- Figure **6.20** remains an unresolved evidence mapping. The manuscript study domain is industrial defect images but the repository candidate `figure_20.png` is a handwritten-digit figure. No PDF examination currently establishes which binary the prior compiled chapter actually included. Do **not** replace Figure 6.20 with a tissue, NCI or digit diagram merely on a filename match.
- The archived Pass44 baseline lists **48 figures, 17 tables, 11 code listings and 14 parent application/case-study units**; the current full production inventory and manuscript assets are absent. Five staged figures are not a full figure audit.
- The proportional scientific ledger remains **12 specifically scoped real-data experiments scientifically certified (11 Level A, 1 Level B)**; this is not certification of all 14 broader parent titles.
- Unverified genuinely required experiments still include multi-scene real-data downstream driving perception for broad synthetic-driving claims and independently measured strength of novel generated concrete recipes if physical performance is asserted.

**Pass61 verdict:** five useful non-plot scientific assets staged and verified locally; actual model evidence and published figure diversity improved; manuscript-wide scientific-parent and final production approvals remain **HOLD**. GitHub workflow for architecture regeneration has been committed, but CI execution is unconfirmed.
