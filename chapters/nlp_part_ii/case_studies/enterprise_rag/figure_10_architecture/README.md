# Figure 10 — Enterprise Retrieval-Augmented Generation Architecture

**Section:** Case Study: Enterprise Retrieval-Augmented Generation  
**Subsection:** Model Architecture  
**LaTeX label:** `fig:rag_architecture`

This remediated figure represents the executable RAG architecture through measured computational states rather than a generic arrow-based flowchart.

The underlying experiment uses an explicit enterprise-policy corpus, TF-IDF unigram/bigram cosine retrieval, deterministic lexical reranking, evidence-bearing prompt assembly, and a grounded response retaining source identifiers.

The panels are:
- **(a)** retrieval similarity spectrum across the enterprise documents, including the top-k threshold;
- **(b)** reranking decomposition for retrieved candidates, separating cosine similarity, query-token coverage, and the combined reranking score;
- **(c)** document-level evidence retention across retrieval, reranking, grounded-prompt assembly, and response citation, shown as a binary quantitative matrix with a 0/1 colorbar.

The figure therefore exposes the architecture through actual intermediate states and evidence retention rather than decorative boxes and arrows. Numerical dimensions, rankings, scores, prompt, and evidence IDs are saved as provenance. No benchmark-performance claim is invented.
