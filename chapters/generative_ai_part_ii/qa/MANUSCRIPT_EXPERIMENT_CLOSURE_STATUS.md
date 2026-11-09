# Chapter 9 — Manuscript and Experimental Evidence Closure

**Date:** 2026-11-09  
**Canonical repository chapter:** Chapter 9, Generative AI Part II  
**Release decision:** **HOLD**; no completed full-chapter certification may be asserted.

## Verified present

- 11 frozen figure specifications decoded from the actual canonical figure generator
- Fail-closed figure and production evidence gates with GitHub-hosted unit-test success
- Manuscript integration audit with GitHub-hosted execution
- Chapter figure-topic and evidence crosswalk at `qa/PRODUCTION_FIGURE_CROSSWALK.csv`

## Verified missing from canonical GitHub main

- `chapters/generative_ai_part_ii/manuscript/` containing final reader-facing LaTeX
- A complete final compiled Chapter 9 PDF
- `chapters/generative_ai_part_ii/figures/` containing 11 final reviewed production figures
- Eleven real-data experiment provenance records with generator source, empirical metrics, figure assets, integrity hashes and checked captions

**GitHub Actions:** the manuscript integration check and figure-inventory/report steps pass; the strict manuscript gate fails as designed. A failing production gate is not completed certification.

## The eleven actual figure topics

1. Empirical Scaling Regimes
2. Compute–Quality Pareto Frontier
3. Scaled Dot-Product Attention
4. KV-Cache Memory Hierarchy
5. Sparse Expert Routing
6. Three Multi-Agent Architectural Patterns
7. Agent Planning and Verification Loop
8. Multimodal Foundation Model Fusion Architecture
9. Latent Video Diffusion Architecture
10. Neural Radiance Field Rendering Pipeline
11. Distributed Transformer Parallelism

Figures 3–11 are predominantly conceptual architecture/pipeline figures in the frozen inventory. The strict **every figure must be based on actual experimental data** requirement therefore needs figures redesigned around measured tensors, traces, performance metrics or outputs from real-input model executions. Invented diagrams, synthetic measurements or inappropriately repurposed Chapter 8 digits experiments do not satisfy this requirement.

## Identified figure-number conflict

`chapters/generative_ai_part_ii/scaling_laws/figure_04_compute_quality_pareto_frontier/` contains a published **analytical fit** to the Chinchilla scaling law, but the canonical frozen inventory lists the **Compute–Quality Pareto Frontier** as **Figure 2**, with Figure 4 referring instead to the KV-cache hierarchy. Neither silently rename nor present this script as empirical Figure 4. The source and caption mappings must be corrected against the recovered manuscript, and real observations added to support an empirical figure.

## Requirements before publishing a final chapter

1. Restore the actual canonical manuscript, bibliography, and stable figure labels to `manuscript/`.
2. Execute and commit the eleven model-/dataset-specific experiments and their measured CSV/JSON, final figure assets, and exact hashes.
3. Match all eleven frozen figure topics, figure labels, captions, nearby discussion and scientific claims to the real outputs.
4. Compile the final PDF, audit visual quality, panel labels, margins and cross-references.
5. Pass GitHub-hosted real-data evidence, manifest, and manuscript integration gates.
6. Review scientific correctness; automated checks alone do not establish data authenticity or figure interpretability.

**No PDF was regenerated, no missing experimental outputs were fabricated, and none of the 11 figures received full production certification in this pass.**
