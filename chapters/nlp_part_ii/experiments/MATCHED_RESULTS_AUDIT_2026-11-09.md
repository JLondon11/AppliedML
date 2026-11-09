# Chapter 9 — Matched experiments, 2026-11-09

This audit documents actual completed evaluations on original public data and frozen predictions. It **does not** certify all eight Chapter 9 case-study units.

## 1. CNN/DailyMail BART versus Lead-3

32 exact same source articles and references verified using original execution artifacts. BART pretrained inference mean ROUGE-L F1 0.2516402; matched Lead-3 mean 0.1991040; mean within-document gain +0.0525363. BART higher on 23/32 articles; exploratory 50k paired bootstrap interval [0.01908,0.08666], sign-test p=0.02006. **First 32 documents are not a randomized or full-split comparison.** Full 11,490-document Lead-3 baseline ROUGE-L 0.2484612 is reported independently, not presented as a paired contrast.

GitHub source: `summarization_matched/compare.py`, `summarization_lead3/run.py`, `seq2seq_evaluation/run.py`.
Original model artifacts are in the GitHub Actions run `42596989928` (BART sampled) and `42599386767` (Lead-3 full).

## 2. WMT14 MarianMT versus copy-source control

All 3,003 test records have matching source strings, references, and ordered IDs. The source/reference SHA-256 recorded in both original executions is `bad8b5bb75fa556be3b8f94e55b2546ab23ab4bf372643594bd7290ca3c64877`.

| Model | Corpus SacreBLEU | Corpus chrF |
|---|---:|---:|
| MarianMT (actual pretrained model predictions, full test) | 33.74766 | 59.82808 |
| Copy original German input (sanity control, full test) | 2.76262 | 24.91310 |

These results are real but the control is intentionally weak; a competitive NMT baseline and an exact model revision pin are still desirable before scientific final certification. MarianMT was evaluated with up to 512 source tokens, max 96 output tokens, 2 beams, checkpoint `Helsinki-NLP/opus-mt-de-en`, revision `main`. The corpus-level BLEU difference is not a per-sentence difference.

GitHub source: `wmt14_matched/compare.py`, `seq2seq_evaluation/run.py`, `wmt14_copy_baseline/run.py`.
Original workflows: `42597710807` and `42711489450`.

## State of other Chapter 9 case studies

The IMDb TF-IDF baseline has actual full official test results of 0.90504 accuracy and macro-F1 0.9050398 over 25,000 held-out reviews. The BERT pilot has actual one-epoch fine-tuning and measured subset metrics, **not** a full BERT benchmark. The SciFact, FinQA, and TAT-QA retrieval/routing experiments have real records and recorded metrics but do not substitute for end-to-end research agent, scientific literature assistant, or authorized enterprise system experiments. DocVQA official page images are still not acquired.

**No invented experiments or synthesized empirical numbers may be used in a Chapter 9 case-study figure.** Scientific completion requires accurate task-model attribution, saved per-example outputs, exact data and model provenance, statistics, genuine figures, and manuscript integration.
