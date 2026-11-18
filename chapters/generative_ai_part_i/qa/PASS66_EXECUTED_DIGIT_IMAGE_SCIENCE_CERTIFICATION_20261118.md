# Chapter 6 — Pass 66: authentic executed digit-diffusion image science and release-quality evidence

**Review date:** 2026-11-18. **Experimental scope:** CH6-D12, the existing *Level A* original 8×8 handwritten-digit class-conditioned latent diffusion diagnostic. **Scientific conclusion:** original executed-model source, original held-out scans and generated tensors verified. This expands the previously certified experiment's empirical visual evidence; it **does not count as a new independent thirteenth certificate** or certify natural-language Stable Diffusion. **Full Chapter 6 production gate remains HOLD** pending the actual complete current PDF/LaTeX.

## 1. Verified actual CI executions, not inferred code success

GitHub Actions workflow `.github/workflows/chapter6_real_tissue_and_digits_scoped_science.yml` succeeded on **run 44209629586** after removing incorrectly imposed archived image counts from genuine reruns. Both jobs succeeded, with the original scanned-digit model run producing 1,797 authentic scanned digit source observations, a leakage-separated **1,257/270/270 train/validation/test split**, actual 25-epoch digit VAE and 35-epoch label-conditioned/unconditional 48-step latent denoisers. The classifier trained on original real scans obtained **95.55555556% accuracy on 270 genuinely held-out original scans**. The actual output archive preserves 1,000 model-generated digit tensors per model arm and **2,000 measured classifier probability records**.

The latest executed run's observed requested-label agreement was **779/1,000** (77.9%) for class-conditioned generator and **94/1,000** (9.4%) for unconditional generator. The older Pass27 archive's **778/1,000 and 94/1,000** is an earlier executed-result record and remains historically valid. Results differ by one sample across separately executed environments; the latest run has not been falsified or forced to match the previous run.

**Provenance:** Source run **44209629586**, original evidence artifact **14384184921**, `chapter6-real-scanned-digits-class-conditional-model-evidence`. Original-source scanned-digit hash `f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443`. Original per-image prediction CSV SHA256 `de376724ba102a09f67e75208aa90dd57b79d2caf81942f19af421feaae4c2d3`. Conditional generated tensor SHA256 `bebaaa56004c1180fc0e6daa09198f5c74e8a527e6cdb4d149ed943eddfe0645`, unconditional generated tensor SHA256 `70f93ba0b05d61965df45a2bf24afb61c4785d8569b94ee8210e42fe19acf708`, original train/validation/test tensor SHA256 `283aa97a11a75a96b7e0f8e57c6101c1ee58e3bb36f3a4b4d593fba549b11974`. This is actual experimental source metadata, **not fabricated or simulated measures**.

## 2. Measured classwise generator error is not uniform

All 2,000 actual saved classifier output probability rows were retrieved, independently parsed, verified for disjoint arm/index keys, normalized probabilities and requested-label alignment. Conditional match counts (100 actual samples per class, digits 0 through 9) were:

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Conditional correct of 100 | 93 | **39** | 55 | 92 | **99** | 67 | 92 | 91 | 67 | 84 |
| Unconditional correct of 100 | 4 | 2 | 3 | 16 | 11 | 12 | 7 | 7 | 25 | 7 |

Thus the aggregate model score hides meaningful weakness for **requested digit 1** (39%) and high agreement for **digit 4** (99%), on this fixed checkpoint. This is classifier-consistency performance, **not independent expert-judged image realism, new observed photographic data or robust performance across independent model-training seeds**.

The small current-run verified source summary is committed in `code/section_6_20_1_latent_diffusion_digits/qa/ci_44209629586/original_executed_class_counts.csv`, with full original artifact and source-tensor SHA256 evidence in `original_evidence_provenance.json`. The underlying individual prediction/generation tensors are preserved in the GitHub Actions artifact and user evidence package; they are **not falsely described as individually committed repository files**.

## 3. Two new genuinely informative scientific image figures — not plots

The new complete figure source `code/section_6_20_1_latent_diffusion_digits/render_actual_generated_vs_original_digits.py` reads:
1. One **real held-out original scanned digit** per class 0–9;
2. The **first actually generated conditional** output for the same requested class;
3. The **first actually generated unconditional** output for the same requested class;
4. The matching actual pretrained-on-real-scans classifier predicted label and probability records.

Selection is deterministic by source index and exact saved model-output row, **not by visually choosing successful examples**. Digit classes 0–4 and 5–9 appear on separate **6.5-inch-wide vector-captioned PDF plates**. Thirty input/model image panels are supported by distinct SHA-256 hashes of their true 8×8 pixel arrays in the figure provenance JSON. Errors are shown openly; for example, the first requested class-1 conditional image is classified as digit 8.

The new figure-generation GitHub Actions run **44423296969** was **confirmed SUCCESS**, producing artifacts **14482777674**, including both PDFs, PNGs, SVGs and an original data/image-pixel provenance record.

**Production-preflight result:** each PDF is **468 pt wide**, has 15 genuine input/model output panels, **8.1 pt minimum selectable vector type**, no off-page text and clean independently inspected rendered-page previews. These are **scientifically verified standalone print-width assets**, not yet verified inserted, numbered or captioned in the latest compiled Chapter 6 manuscript.

Actual original PDF SHA256 of first successful CI build:
- Page 1: `9cefed2b47fb80053aee0688a3eee090921e26065efd46fdf197e1f4e3b628a5`.
- Page 2: `0a94bd571d76d253c2bacc1c99c05b974a9d3b48bbc5b8799ea271e2dbf81939`.

The corresponding **rendered-page pixel** SHA256 values were `e8041857f9e8f1542ba3044c1a5c5db8b00c4a77a4de9c9aa0f75b9f71ee3c6e` and `5105e0862c65c9c4bc1f1a527a24a14dde3e592b8793221ba39b66bef2f4c9de`. These pixel hashes, not PDF container-byte hashes, should control across repeated PDF builds because generated PDF creation metadata changes file SHA even when the printed pixels remain identical.

A newly committed independent `qa/verify_pass66_actual_digit_image_evidence.py` checks source tensors and original split identities, 2,000 classifier outputs, 30 actual displayed image pixel hashes, original printed figure image count, legibility and matching rendered-page pixel fingerprints.

## 4. True R1–R10 interpretation at the appropriate level

- **R1:** true `sklearn.load_digits()` source-scan provenance, 1,797 original observations and original SHA.
- **R2:** real split 1,257/270/270 disjoint source IDs, with original training-only classifier evaluation.
- **R3:** 25-epoch original-digit VAE, conditional/unconditional 35-epoch diffusion models, actual saved tensors from successful CI.
- **R4:** 2,000 actual model-produced samples and saved prediction probability rows; 30 selected tensors independently source-linked into the real-image scientific plates.
- **R5:** meaningful unconditional model arm and original held-out real-image-trained classifier; not an image fidelity benchmark or matched natural-image generative architecture comparator.
- **R6:** measured overall and per-class correct counts, actual sample denominators and explicitly acknowledged single-seed fixed-generator dependence.
- **R7:** sound precisely for tiny grayscale class-conditioned handwritten digits only, no false Stable Diffusion or clinical/general image claims.
- **R8:** original real image versus true model output figure pages, not only plots, metrics or decorative architecture diagrams.
- **R9:** complete model implementation, generated outputs, source audit and source/GitHub Actions run/artifact IDs; final PDF production is a **separate** gate.
- **R10:** negative class-specific failures, actual original test imagery versus models, classifier proxy limitation and no claim of broader clinical or natural-language quality.

**Scientific decision:** original CH6-D12 remains **Level A SCIENTIFIC_CERTIFIED within its documented scope**; no application-title inflation.

## 5. Production and repository delivery boundaries

The complete figure-generation source, independent scientific verification script and original 22-row current-run class table are committed to GitHub. The original source/model CI artifacts are separately downloadable. A follow-on binary-asset persistence workflow is being validated; its GitHub commit must be independently confirmed before claiming the generated PDF/PNG/SVG are in the repository.

**Publication-ready suggested caption** (no inferred figure number): *Actual handwritten-digit source observations and outputs of the trained class-conditioned and unconditional latent-diffusion generators. Each panel column displays one genuine held-out 8×8 handwritten-digit scan and the first saved neural-network-generated image for that requested class under each generator. Classifier predictions come from a model evaluated on separate genuine handwritten-digit observations; all 2,000 generated-image predictions yield 779/1,000 correct requested-label agreements for conditional generation versus 94/1,000 for unconditional generation. The figure is an original-model diagnostic, not an independent image realism evaluation, and does not represent text-to-image Stable Diffusion.*

**Final Chapter 6 status:** Twelve scoped certifications remain; the broader fourteen named application/case-study titles are still independently reviewed. The current complete 109-page manuscript PDF/LaTeX is unavailable, so full Chapter 6 production approval and the exact manuscript Figure 6.20 binary mapping remain **HOLD**.
