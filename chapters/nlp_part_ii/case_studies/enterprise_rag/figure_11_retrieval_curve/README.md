# Figure 11 — Retrieval Quality as a Function of Top-k Depth

**Section:** Case Study: Enterprise Retrieval-Augmented Generation  
**Classification:** Case Study  
**Subsection:** Results and Error Analysis  
**LaTeX label:** `fig:retrieval_curve`

**Frozen detailed caption:** Retrieval-performance curve showing how evidence recall changes as the number of retrieved documents, k, increases. The figure should reveal the tradeoff between improved evidence coverage at larger k and increased context noise, latency, and prompt cost. It should support the discussion of retrieval-depth tuning by identifying the region where marginal recall gains begin to diminish and where excessive retrieval can degrade downstream generation despite higher raw document coverage.

**Unified production caption:** Retrieval quality as a function of top-k depth on the BEIR SciFact test set using executable TF-IDF unigram/bigram cosine retrieval against the released relevance judgments. **(a)** Mean evidence recall across judged test queries versus retrieval depth k; the band shows ±1 standard error across queries. **(b)** Marginal gain in mean recall as k increases, exposing the region of diminishing returns in additional evidence coverage. **(c)** Mean retrieved text burden and the fraction of retrieved documents not judged relevant versus k, quantifying the context-noise and prompt-cost pressure that grows with retrieval depth. Together the panels show why larger k can improve raw evidence coverage while simultaneously increasing irrelevant context and context-window burden; downstream generation quality is therefore not implied to improve monotonically with recall. Retrieval latency is measured and retained in provenance because it is runner-dependent rather than encoded as a visually misleading universal curve.

Data: public BEIR SciFact test corpus/qrels. The BEIR project documents SciFact as a 5K-document, 300-query retrieval benchmark and provides the public dataset URL. No retrieval metric in the figure is invented.
