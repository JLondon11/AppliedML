# Chapter 9: Real-data model-specific experiments

These scripts train named models against real data. They never synthesize predictions or empirical scores. **Committing code is not scientific certification.** Both experiments require the named external datasets and pretrained checkpoints before a training run is possible.

## BERT sentiment classification on IMDb

Requires `torch`, `transformers`, `datasets`, `numpy`, `scikit-learn`, `matplotlib`. Example:

```bash
python imdb_bert/run.py --checkpoint bert-base-uncased --epochs 3 --batch-size 16 --output results/imdb_bert
```

Downloads the official `stanfordnlp/imdb` 25k/25k dataset, partitions official train deterministically into 22,500 train and 2,500 validation, fine-tunes the named BERT checkpoint, selects best validation macro-F1, and evaluates the **untouched 25,000 test records**. Results include per-example prediction JSONL, loss curves, confusion matrix, split hashes and numerical metrics. For final scientific claims, pin and document the exact model commit/revision.

## LayoutLMv3 document question answering

Requires original licensed DocVQA page images, questions, answers, and OCR words with bounding boxes normalized to 0..1000, plus a pretrained LayoutLMv3 checkpoint. No raw DocVQA data are redistributed. Provide three JSONL manifests with disjoint **document IDs** (not only question IDs). Each line requires `id`, `document_id`, `image` (existing local image), `question`, `words`, `boxes`, `answers` (list of accepted reference strings).

```bash
python docvqa_layoutlmv3/run.py --train data/train.jsonl --validation data/valid.jsonl --test data/test_labeled.jsonl --output results/docvqa
```

The runner fine-tunes LayoutLMv3's extractive QA head using OCR-aligned answer spans and measures held-out ANLS and exact match against multiple references. It records how many training questions cannot be aligned to OCR text (an extractive-model limitation) and writes the actual individual predictions and a validation curve.

**DocVQA caveat:** The official challenge's hidden test annotations are not public. Never present results from a locally held-out labeled evaluation split as official hidden-test leaderboard scores. Ensure no document overlap across training, validation, and labeled held-out evaluation. Pin dataset and checkpoint revisions.

## Scientific certification

A complete experiment requires actual dataset and model execution logs, exact model and dataset versions, train/validation/test IDs, checkpoints, run seed, measured results, baseline comparisons, full test predictions, and figures that can be regenerated from those predictions. Input-contract tests and the earlier manuscript retrieval pilot cannot substitute for the named model executions.

**2026-11-08 execution status:** The current environment has PyTorch but does not have the IMDb and DocVQA data, pretrained BERT/LayoutLMv3 weights, or the Transformers/Datasets packages. External dataset and model hosts are inaccessible here. Therefore, **neither of these two experiments has yet been executed or scientifically certified**.
