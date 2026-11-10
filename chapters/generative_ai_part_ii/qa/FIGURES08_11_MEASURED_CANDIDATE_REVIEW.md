# Chapter 9 — Figure 8 and Figure 11 empirical candidate review

## Figure 8: Image-to-digit-word alignment

A real-dataset-trained dual-encoder experiment was executed on the 1,797-image scikit-learn handwritten-digits dataset (1,257 train / 270 validation / 270 independent test). Source: `code/figure08_digit_word_alignment/experiment.py`. The image encoder maps 64 measured pixel values to 32-dimensional features; a character-word encoder maps the **observed human digit category names** to corresponding vectors, with a supervised class-contrastive loss. The selected epoch is 19 of 60; the independent test achieves 262/270 correct digit-name matches (**97.037%**). Actual PNG/SVG, numerical metrics, per-example predictions, history, and checkpoint are stored as GitHub Actions outputs. The hosted execution succeeded at https://github.com/JLondon11/AppliedML/actions/runs/42775157981 .

**HOLD for canonical architecture certification.** This is not a pretrained CLIP-style natural image-caption alignment model. There are no real paired audio observations, no vision-language-audio fusion mechanism and no retrieval benchmark. Although the model uses real images with human-assigned names, this evidence cannot prove the chapter's broad "multimodal foundation model fusion architecture" claim. It is a narrowly labeled pedagogical experiment only.

## Figure 11: Distributed Transformer parallelism

A genuine 2-head, one-layer Transformer was trained on real digit-row sequences using PyTorch CPU/Gloo data parallelism with **one and two processes**, fixed disjoint 1,257/270/270 partitions. The two configurations used the same global-batch target and four epochs, and both produced independently evaluated held-out test predictions. Source: `code/figure11_real_ddp/experiment.py`. The hosted run succeeded at https://github.com/JLondon11/AppliedML/actions/runs/42776326465 .

| Hosted measured quantity | 1 CPU rank | 2 CPU ranks |
|---|---:|---:|
| Processed training samples per second (including sampler padding) | 22,561.70 | 22,370.55 |
| Measured wall time (seconds) | 0.2229 | 0.2249 |
| Independent held-out test accuracy | 60.74% | 59.63% |

**HOLD for canonical full parallelism certification.** The frozen Figure 11 topic includes tensor, pipeline and other parallelism methods. A two-rank CPU data-parallel benchmark validates only gradient-synchronizing data parallelism. On this small example there is no meaningful measured speedup; the low test accuracy and sampler padding are disclosed. The throughput measurement is processed (possibly padded) samples divided by wall time, not unique samples per second.

## Release decision

- Figures **1–5**: machine-verifiable, hash-checked real-experiment provenance; full manuscript acceptance pending.
- Figures **6–7**: logged real-data tool-orchestration outputs; **autonomous-agent benchmark certification HOLD**.
- Figure **8**: real-data image-label contrastive alignment; **multimodal foundation-model fusion certification HOLD**.
- Figures **9–10**: no trained real-video diffusion or calibrated real-view NeRF experiment committed; **HOLD**.
- Figure **11**: measured real-data CPU DDP experiment; **all-strategy distributed transformer certification HOLD**.
- **Final reader-facing Chapter 9 manuscript remains HOLD:** the canonical LaTeX/PDF assets have not been committed, so final captions, placement, citations, and visual print-scale inspection remain unverified.

No new figure should be counted as fully production-certified merely because its experiment code ran or its artifact is present.
