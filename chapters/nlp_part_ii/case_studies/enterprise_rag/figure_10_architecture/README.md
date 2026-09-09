# Figure 10 — Enterprise Retrieval-Augmented Generation Architecture

**Section:** Case Study: Enterprise Retrieval-Augmented Generation  
**Subsection:** Model Architecture  
**LaTeX label:** `fig:rag_architecture`

The frozen caption requires the complete query-to-grounded-response path,
explicit evidence retention, and a distinction between parametric model
knowledge and non-parametric enterprise knowledge.

This production artifact is instantiated by executable retrieval code over a
small explicit enterprise-policy corpus. The code computes a TF-IDF
unigram/bigram representation, cosine retrieval, top-k=3 selection,
deterministic lexical reranking to k=2, evidence-bearing prompt assembly, and a
grounded response retaining source identifiers. Numerical tensor dimensions,
rankings, scores, prompt, and evidence IDs are saved as provenance.

The artwork is a scientific computational/data-flow rendering, not a decorative
infographic. It contains no overall title or caption and makes no invented
benchmark-performance claim. Freshness, chunking/retrieval, ranking, and
grounding/citation retention are located at the stages where they can affect
generation quality and hallucination risk.
