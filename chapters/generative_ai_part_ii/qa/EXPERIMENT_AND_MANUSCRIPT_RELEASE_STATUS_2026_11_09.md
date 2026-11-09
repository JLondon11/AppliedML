# Chapter 9 — Measured Figure Evidence and Final Manuscript Status

**Canonical chapter:** 9 — Generative AI Part II  
**Audit date:** 2026-11-09  
**Evidence milestone:** **5/11 measured real-data figure assets verified**  
**Final reader-facing manuscript production certification:** **HOLD (0/11 fully certified in the manuscript)**

## Experimental evidence committed to GitHub

The real-data experimental implementations and their GitHub Actions workflows have been executed, and the measured outputs for the first five frozen figure identifiers have been committed to `chapters/generative_ai_part_ii/figures/`. The canonical figure generator is now a hash-checked dispatcher; generic synthetic renderers cannot replace a missing experimental figure.

| Frozen figure | Experimental interpretation | Dataset; train / validation / independent test | Status |
|---|---|---|---|
| 1. Empirical Scaling Regimes | Five MLP widths × three training seeds; actual held-out cross-entropy | sklearn real handwritten digits; 1257 / 270 / 270 | **Experimental evidence verified** |
| 2. Compute–Quality Pareto Frontier | Same 15 measured training runs; FLOPs proxy is calculated, CPU training time is measured | sklearn real handwritten digits; 1257 / 270 / 270 | **Experimental evidence verified** |
| 3. Scaled Dot-Product Attention | Measured Q/K/V scaled scores and attention weights from trained row-token classifier | sklearn real handwritten digits; 1257 / 270 / 270 | **Experimental evidence verified** |
| 4. KV-Cache Memory Hierarchy | Measured full-versus-cached inference equivalence, memory formula and CPU latency | sklearn real handwritten digits; 1257 / 270 / 270 | **Experimental evidence verified** |
| 5. Sparse Expert Routing | Trained top-2 MoE routing and expert load on held-out images | sklearn real handwritten digits; 1257 / 270 / 270 | **Experimental evidence verified** |
| 6–11 | Agents, agent loops, multimodal fusion, video diffusion, real-view NeRF, distributed transformer | Real model-/dataset-specific evidence not yet committed | **HOLD** |

All five models were trained with real handwritten-digit images. **These are compact pedagogical experiments, not trained frontier LLMs, long-context production servers, large-scale MoE systems, or published language-model scaling evaluations.** The figures can support narrowly described mechanisms and measured small-model results; they must not be used to imply verification of foundation-scale claims. Particularly, the scaling plots do *not* validate Kaplan or Chinchilla fits, and the KV-cache CPU microbenchmark does not support a general latency-speedup claim.

## Machine-verified versus manuscript-certified

**The GitHub-hosted Chapter 9 evidence audit reports 5/11 figures passing machine-verifiable provenance and integrity checks.** These checks establish that the referenced code, measured results, dataset specification, train/validation/test split counts, and final asset hashes are present and consistent. They are not a substitute for a scientific peer review of the manuscript interpretation.

The canonical final `manuscript/` LaTeX source, bibliography and compiled final PDF remain absent from the repository. Therefore the following claims **cannot** be made: that the final captions are correct and integrated; every figure is introduced and discussed appropriately; the actual main text has all citations, numbered equations and references; all floats have acceptable placement; or that the final PDF was compiled and inspected. These remain **HOLD** for all eleven figures.

## Remaining experimental figure requirements

- Figure 6: Execute and measure actual agent configurations on a named real benchmark, including runs, errors and outcomes.
- Figure 7: Capture genuine agent planning/tool/verification execution traces and benchmark outcomes.
- Figure 8: Train or evaluate the actual named multimodal components with real paired image/text/audio sources as applicable; report quantitative alignment.
- Figure 9: Execute model-specific latent video generation on an actual video dataset, including real input frames and evaluated output.
- Figure 10: Use calibrated multi-view real imagery, train/evaluate a NeRF, and report novel-view renderings and metrics.
- Figure 11: Use real distributed-training hardware/configuration traces and measured communication, memory, throughput and scaling effects.

Once real experiments are available, synchronize the true manuscript and bibliography, conduct a section-by-section scientific/citation review, compile its PDF, and verify captions, panel labels, equation references, and placement. **Do not declare final publication approval solely from the 5/11 machine-readable evidence gate.**

## Source of truth

- `qa/figure_evidence.json` — machine-readable figure-specific evidence, hashes, and caption candidates
- `qa/PRODUCTION_FIGURE_CROSSWALK.csv` — per-figure status
- `qa/audit_figure_evidence.py --production` — strict real-data/integrity validation
- `qa/audit_manuscript_integration.py` — final manuscript structural gate
- `figure_generation/generate_figures.py` — canonical SHA-verified real-data figure renderer

**Decision: 5 experimental figure records VERIFIED; remaining 6 HOLD; final manuscript release HOLD.**
