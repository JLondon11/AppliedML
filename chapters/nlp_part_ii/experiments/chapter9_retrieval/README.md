# Chapter 9: manuscript-grounded retrieval audit

This directory contains a post-hoc analysis of the 39 author-constructed questions and saved rankings from the Chapter 9 experimental execution package. The corpus comprises PDF-extracted Chapter 9 manuscript text; this is **not** an independent public benchmark and is **not** a trained generative RAG experiment.

## Running

Install Python, NumPy. Copy these frozen input files from the accompanying Chapter 9 real-data execution package into this directory:

- `source/chapter9_extracted_text.txt`
- `fixed_queries.csv`
- `sensitivity_query_results.csv`
- `extractive_evidence_traces.csv`

Run `python shortcut_evidence_audit.py`. It creates `shortcut_audit_results.json` with grouped top-1 retrieval results, Wilson intervals, provenance checksum, and possible extractive-text quality flags. No model or network access is required.

## Measured outcomes (39 questions)

Original saved top-1 correct counts: BM25 22/39, word TF-IDF 21/39, word+character TF-IDF 26/39, title-only TF-IDF 20/39. Among questions with no content-term overlap with the gold heading (11), word+character TF-IDF retrieves 4/11 correctly. With some overlap (28), it retrieves 22/28 correctly. Evidence quality flags are diagnostic only and are not verified errors.

## Certification limits

The 39 questions were not independently labeled and the subset analysis is post hoc. A source-matching quotation is not necessarily relevant, true, or a correct answer to the question. No case study can be certified from this pilot. The eight required Chapter 9 application/case-study experiments still need protocol-matched external datasets, exact model checkpoints, recorded training/inference, predictions, and verified results figures. Never label manuscript-derived retrieval results as an IMDb, DocVQA, translation, summarization, SciFact, or enterprise RAG benchmark.
