# Pass 16 — Priority Scientific Figure Remediation

## Scope

This pass executes asset-level remediation for the highest-priority figures identified in Pass 15. Final ACCEPT requires source/provenance reconciliation, successful regeneration, and visual inspection of the rendered asset.

## Accepted after asset-level inspection

- **NLP II Figure 4 — ACCEPT.** Real BERT-Tiny contextual embeddings, PCA axes with explained variance, and original-space centroid cosine distances with a quantitative colorbar.
- **NLP II Figure 6 — ACCEPT.** Aggregate multi-stimulus layer-by-head routing diagnostics replace fragile single-example attention maps. Four heatmaps share an external quantitative colorbar; the caption explicitly avoids causal interpretation.
- **NLP II Figure 7 — ACCEPT.** The former arrow-chain architecture is replaced by measured tensor/state diagnostics from an IMDb-fine-tuned BERT-base forward pass: token-by-hidden activations with colorbar, layer-wise [CLS] evolution, and trained classifier output.
- **NLP II Figure 8 — ACCEPT.** Shared BERT-Tiny/IMDb experiment with 2,000 train, 500 validation, and 1,000 held-out test reviews. Cells report counts and row-normalized percentages with a quantitative held-out-review colorbar. Provenance reports TN=410, FP=104, FN=262, TP=224, test accuracy 0.634, F1 0.5504.
- **NLP II Figure 9 — ACCEPT.** Training/validation cross-entropy comes from the same shared experiment as Figure 8 and explicitly shows selected-checkpoint logic.
- **Scientific AI Figure 14 — ACCEPT as methodological demonstration.** Real RetinaMNIST held-out image, executable CNN, computed Grad-CAM, and quantitative 0–1 activation colorbar. The caption limits interpretation to a low-resolution methodological benchmark rather than clinical-grade attribution.

## NLP II Figure 12 — ACCEPT

The claim-level SciFact design has been revised, regenerated from the committed 120-claim result table, and visually inspected. The mean paired change (RAG minus standalone unsupported-term fraction) is approximately -0.052 with a bootstrap 95% interval approximately [-0.112, 0.010]. Because the interval spans zero, the manuscript does not claim a definitive reduction under this protocol.

The accepted design uses:
- claim-level standalone vs retrieval-grounded scatter;
- dashed identity line;
- numerical paired mean-change and bootstrap-interval annotation;
- residual retrieval-miss vs generation-failure decomposition.

The dedicated render-only workflow completed successfully and the final PNG/SVG were reinspected after commit.

## Workflow hardening

- Scientific AI Figure 14 is owned by its dedicated workflow.
- NLP II Figure 12 is owned by its dedicated workflow.
- The consolidated remediated-figure workflow handles NLP II Figures 3, 4, 6, and 10.
- Consolidated regeneration is serialized with workflow concurrency.
- Generic source remediation is never treated as final ACCEPT without rendered-output inspection.

## Book-wide gates retained

- 415 / 415 canonical figure environments.
- 15–22 substantive reader-facing code listings per chapter.
- All heatmaps/matrices/saliency fields require a quantitative legend or colorbar.
- Generic deterministic surrogate plots cannot satisfy captions requiring an exact method, dataset, benchmark, simulator, or empirical computation.
- Exact-duplicate figure audit remains a release gate.
