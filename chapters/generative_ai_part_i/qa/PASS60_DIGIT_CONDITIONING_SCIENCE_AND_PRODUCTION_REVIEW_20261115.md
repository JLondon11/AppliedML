# Chapter 6 — Pass 60: scientific class-conditioning evidence and production handoff

**Date:** 2026-11-15.  
**Book-wide standard:** `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md` (proportional R1–R10; true observed data, executed named model, actual outputs, source-linked figures, truthful claims, and distinct manuscript-production certification).

## Reopened source and verification

The user-supplied `Proceed Chapter 6.txt` is a **Pass 37 handoff statement**, not the actual chapter PDF/TeX/image. It reports that Figure 6.20 was integrated with its caption/discussion on printed page 67 and the end-of-chapter problems start on page 101. That is valid evidence of what the earlier production pass *reported*; it is **not direct new verification of the current final source/asset mapping**.

The current GitHub chapter root still lacks a current `manuscript/` directory, and `quality_assurance/chapter6_evidence_manifest.json` explicitly records Figure 6.20 as unresolved, with a misleading candidate from a handwritten-digit script and no verified image SHA. Do not assume that legacy `figure_20.png` is the actual industrial-defect publication figure.

The previous actual executed production-QA archive `Chapter6_Pass59_Executed_Production_QA.zip` was reopened; `python -m unittest discover -s qa -p 'test_ch6*.py' -v` executed **21 tests, 21 passes**, independently repeating the archived run. These are *software fixture tests*, not real experimental reruns.

## New scientific audit: CH6-D12 class-conditioned real-digit latent diffusion

Verified the exact GitHub source `code/section_6_20_1_latent_diffusion_digits/figure_6_44_class_counts_pass27.csv`, Git blob SHA-1 `ed1d87dfdd570131278044b1b51df19cad55b3af`. It has 22 records: 10 per-digit records per arm plus an overall record for each arm. All 20 individual digit rows have exactly 100 generated records; per-digit correct counts sum to the original recorded totals:

| Original model-output diagnostic | Conditional generated | Unconditional generated |
|---|---:|---:|
| Classifier agrees with comparison label | 778 / 1000 = **77.8%** | 94 / 1000 = **9.4%** |
| Digit 1 class agreement | 39 / 100 = **39%** | 2 / 100 = 2% |
| Digit 2 class agreement | 55 / 100 = **55%** | 3 / 100 = 3% |
| Digit 4 class agreement | 99 / 100 = **99%** | 11 / 100 = 11% |

Other conditional classes below 70% are digit 5 (67/100) and digit 8 (66/100). The class range 39%–99% demonstrates meaningful heterogeneity hidden by the 77.8% aggregate. The unconditional arm has **assigned comparison labels**, not genuine conditioning prompts; its low same-label match rate is only an indirect reference. One trained generator seed and automated digit-classifier scoring cannot independently certify natural image fidelity, broad diffusion superiority, semantic human realism, or Stable Diffusion text-to-image performance.

A new SVG, `code/section_6_20_1_latent_diffusion_digits/figures/verified_digit_conditioning_science.svg`, was generated **directly from these existing numerical source records**. It combines (A) measured class-conditional vs unconditional agreement using paired dot-and-line comparisons (no wide bar charts), with (B) a faithful conceptual diagram of the actual original-digits / train/val/test / 16D VAE / 48-step latent denoiser / original-trained classifier pipeline. The conceptual panel is methodology, not invented numerical evidence. The full generator is `plot_verified_conditioning_pipeline.js` in that section. The CI workflow has a regeneration+diff test; CI execution was not independently confirmed.

**These are a new examination and figure, not a newly trained model.** Existing CH6-D12 Level A certification remains strictly within the prior actual digit-generation diagnostic scope. The figure is a **scientific publication candidate**, not yet assigned a page/figure number in the current manuscript.

## Production QA / manuscript hold

The archived Pass 44/45 inventory reports 109 pages, 48 figures, 17 tables, 11 numbered code listings and 14 applications. Until the actual current manuscript assets are accessible, full production QA cannot certify:
- exact figure-to-caption/source correspondence or Figure 6.20 publication file;
- correct scientific placement of the new digit figure or avoidance of superseded legacy figures;
- table accuracy/relevance and figure visual inspection *in the compiled PDF*;
- reference resolution, text quality, code listing completeness and absence of narrative production notes;
- complete title-to-claims matching for all fourteen parent application/case-study units.

Current live register records **12 separately and narrowly scoped scientific certificates** (11 Level A, 1 Level B), all production HOLD. It is false to equate these with 12/14 parent applications certified. Broad synthetic-driving benefit remains gated on a true multi-scene BDD100K/downstream perception result, and **physical strengths of novel generated concrete** remain unmeasured.

**Decision:** Per-class scientific audit VERIFIED from original committed summary, real-source-driven scientific figure/code/CI committed, archived QA tests independently re-executed (21/21). New figure manuscript integration and full chapter scientific/production approval are **HOLD**; no unsupported parent application promoted.
