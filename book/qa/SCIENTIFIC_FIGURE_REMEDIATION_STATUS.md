# Scientific Figure Remediation Status

This status follows the figure-by-figure optimality audit.

## Source remediations completed

| Figure | Audit status | Source remediation |
|---|---|---|
| NLP II Figure 3 — Byte-Pair Encoding | REVISE | Actual merge operations are now exposed directly on the merge-frequency axis so the reader sees which pair is merged at each step. |
| NLP II Figure 4 — Embedding Geometry | REVISE | PCA axes now report explained variance; the interpretive centroid-arrow panel is replaced by a quantitative cosine-distance matrix computed in the original 128-D representation space with colorbar. |
| NLP II Figure 6 — Attention Head Specialization | REPLACE | Single-stimulus selected-head matrices are replaced by layer-by-head mean routing-score maps aggregated across multiple controlled stimuli for positional, syntactic, semantic, and induction-like diagnostics. |
| NLP II Figure 10 — Enterprise RAG Architecture | REVISE | Generic arrow-flow rendering is replaced by measured retrieval spectrum, reranking decomposition, and evidence-retention state matrix. |
| NLP II Figure 12 — Hallucination Reduction | REPLACE | Aggregate condition bars are replaced by paired claim-level measurements with bootstrap summary intervals plus residual failure decomposition. |
| Scientific AI Figure 14 — Grad-CAM Retinal Application | REVISE | Quantitative normalized Grad-CAM colorbar added; documentation now explicitly limits interpretation to a low-resolution methodological benchmark demonstration. |

## Render-state warning

The executable figure-generation sources and documentation have been remediated. Existing committed PNG/SVG files may still represent the previous rendering until the updated scripts are executed and the generated assets are recommitted.

Therefore these figures are **SOURCE REMEDIATED / RENDER PENDING**, not final ACCEPT, until:
1. the updated scripts execute successfully;
2. PNG/SVG assets are regenerated;
3. heatmap/colorbar, panel labeling, spacing, typography, and caption correspondence are visually inspected;
4. duplicate-figure QA is rerun;
5. the final assets pass the Scientific Figure Optimality standard.

No figure should be marked final ACCEPT solely from source-code remediation.
