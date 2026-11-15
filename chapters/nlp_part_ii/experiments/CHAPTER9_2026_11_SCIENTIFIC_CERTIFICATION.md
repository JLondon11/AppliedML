# Chapter 9 — Scientific certification decision (2026-11-15)

This decision is based on the edited **98-page archived-source proof**, *not* the newer 133-page chapter, whose matching source is unavailable in the current materials. Publication Master QA remains **HOLD**.

## Defined-scope scientific acceptance

**Application: Machine Translation Quality Evaluation (archived section 9.29)** — ACCEPT for *automatic reference-based evaluation on the full official WMT14 de-en test split only*. Real pretrained `Helsinki-NLP/opus-mt-de-en` and a copy-source sanity control produced matching source/reference fingerprints across all 3,003 test pairs. MarianMT scores **33.7477 SacreBLEU, 59.8281 chrF**, versus control **2.7626, 24.9131**. Outputs and scripts: `seq2seq_evaluation/run.py`, `wmt14_copy_baseline/run.py`. No COMET/MQM human comparison was run.

## Verified but limited component evidence

- **BERT/IMDb sentiment (archived 9.21)**: full official 25,000-test TF-IDF logistic regression baseline accuracy **0.90504**, macro-F1 **0.90504**, and independent 600-train/500-test BERT pilot accuracy **0.688**, macro-F1 **0.65978**. Not comparable training budgets; full BERT run missing. See `imdb_baseline/` and `imdb_bert/`.
- **Summarization (archived 9.30)**: full 11,490-test CNN/DailyMail Lead-3 ROUGE-L **0.24846**; BART/Lead-3 first-32 paired ROUGE-L **0.25164 vs 0.19910**. BART full-test and factual faithfulness checks absent. See `summarization_lead3/`, `seq2seq_evaluation/`, `summarization_matched/`.
- **Scientific literature assistant (archived 9.23)**: SciFact 5,183-document corpus and held-out annotated evidence retrieval; no generated-answer support/citation verification.
- **Enterprise knowledge assistant (archived 9.24)**: real FinQA/TAT-QA component evidence, but no access-control or end-to-end enterprise answering validation.

## Not certified

- **Enterprise RAG (9.22)**: no authorized corpus, grounded generated answers, or policy checks.
- **Multimodal document intelligence (9.25)**: DocVQA page images/checkpoint inference absent.
- **Autonomous research agent (9.26)**: no end-to-end tool-action traces or agent benchmark.

The application and case-study claims must be scoped to exactly what was executed. A public corpus, a plotting script, or a retrieval metric alone does not certify a generator or an enterprise security workflow.

The archived proof corrections also add two measured-results tables, correct incompatible translation/summarization study descriptions, preserve 11 actual figure assets, and retain 63 numbered-and-referenced equations. Source and PDF accompany the local scientific remediation artifact, not this repo file.
