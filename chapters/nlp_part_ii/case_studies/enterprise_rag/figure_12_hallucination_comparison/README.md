# Figure 12 — Hallucination Reduction Through Retrieval Grounding

**Section:** Case Study: Enterprise Retrieval-Augmented Generation  
**Classification:** Case Study  
**Subsection:** Results and Error Analysis  
**LaTeX label:** `fig:hallucination_comparison`

**Unified production caption:** Hallucination reduction through retrieval grounding in a controlled SciFact experiment using the same FLAN-T5-small generator in both conditions and executable TF-IDF retrieval against the released SciFact corpus and relevance judgments. **(a)** Paired claim-level unsupported-term fractions for standalone generation and retrieval-grounded generation. Each line joins the two measurements for the same evaluated claim, while the larger summary markers and error bars show condition means with 95% bootstrap confidence intervals; this makes both the aggregate change and heterogeneous claim-level behavior visible. **(b)** Residual failure incidence for the retrieval-grounded condition, separating top-1 retrieval misses from generation failures that remain after a relevant document is successfully retrieved. Unsupported content is operationalized reproducibly as the fraction of non-stopword answer terms absent from the released gold evidence documents. The lexical support metric is an automatic proxy rather than human factuality adjudication, so the values are protocol-specific and are not presented as state-of-the-art benchmark scores.

No title or caption is embedded in the artwork. Panel identifiers are below the panels. Claim-level results, paired changes, bootstrap intervals, failure categories, and full provenance are retained in the repository.
