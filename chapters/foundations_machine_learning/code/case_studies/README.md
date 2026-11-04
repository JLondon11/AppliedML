# Chapter 1 — Real-data case studies and applications

This directory contains the complete executable implementations for the empirical
case studies/applications in **Foundations of Machine Learning**. Each script
produces numerical metrics, per-example predictions, provenance metadata, and
publication-ready SVG/PNG result figures.

## Implementations

- `medical_diagnosis_wdbc.py` — Wisconsin Diagnostic Breast Cancer: logistic
  regression and random forest; ROC and confusion-matrix outputs.
- `weather_seattle.py` — Seattle daily weather: chronological next-day
  precipitation classification/regression; ROC and forecast-trace outputs.
- `edge_digits_quantization.py` — handwritten digits: full-precision versus
  symmetric int8 logistic-regression coefficients; accuracy/storage outputs.
- `aerospace_cmapss_fd001.py` — NASA C-MAPSS FD001: engine-disjoint RUL
  regression; scatter and held-out-engine trajectory outputs.

## Reproduction

From the repository root:

```bash
python chapters/foundations_machine_learning/code/case_studies/medical_diagnosis_wdbc.py
python chapters/foundations_machine_learning/code/case_studies/weather_seattle.py
python chapters/foundations_machine_learning/code/case_studies/edge_digits_quantization.py
python chapters/foundations_machine_learning/code/case_studies/aerospace_cmapss_fd001.py
```

Each script accepts `--output-dir` to redirect artifacts. The weather and
C-MAPSS scripts download the cited public datasets at runtime; the medical and
digits datasets are distributed with scikit-learn.

## Master-QA provenance rule

Empirical figures are acceptable only when the manuscript identifies the
dataset, split, model, metrics, and corresponding code path. Simulated or
synthetic outputs must not be presented as case-study evidence when a real-data
claim is made.
