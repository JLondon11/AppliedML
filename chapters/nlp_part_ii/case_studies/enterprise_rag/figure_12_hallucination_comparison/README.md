# Figure 12 — Hallucination Reduction Through Retrieval Grounding

**Section:** Case Study: Enterprise Retrieval-Augmented Generation  
**Classification:** Case Study  
**Subsection:** Results and Error Analysis  
**LaTeX label:** `fig:hallucination_comparison`

**Frozen requirement:** quantitative comparison of unsupported-claim rates for standalone generation versus retrieval-grounded generation, while retaining residual RAG failures caused by retrieval misses, stale/irrelevant evidence, or generation failure.

**Unified production caption:** Hallucination reduction through retrieval grounding in a controlled SciFact experiment using the same FLAN-T5-small generator in both conditions and executable TF-IDF retrieval against the released SciFact corpus and relevance judgments. **(a)** Mean unsupported-content rate for standalone generation and retrieval-grounded generation; unsupported content is operationalized reproducibly as the fraction of non-stopword answer terms absent from the released gold evidence documents, and error bars are 95% bootstrap confidence intervals across evaluated claims. **(b)** Residual failure incidence for the retrieval-grounded condition, separating top-1 retrieval misses from generation failures that remain after a relevant document is successfully retrieved. The experiment therefore tests whether grounding reduces unsupported output while making explicit that RAG can still fail when retrieval is wrong or when generation does not remain supported by available evidence. The lexical support metric is an automatic proxy rather than human factuality adjudication, so the plotted values are protocol-specific and are not presented as state-of-the-art benchmark scores.

No title or caption is embedded in the artwork. Panel identifiers are below the panels.
