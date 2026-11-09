# Chapter 6 — Precision medicine: real cytology CVAE diagnostic (Pass 26)

**Task:** Test conditional VAE generation of genuine Wisconsin Diagnostic Breast Cancer cytology feature records for training augmentation. The data source is `sklearn.datasets.load_breast_cancer()`, originally from Wisconsin Diagnostic Breast Cancer. It has 569 real measurement records, 30 numeric features and benign/malignant labels. This is **one modality**, not multiomics, EHR/imaging fusion, or treatment-response prediction.

**Protocol:** fixed stratified source split seed 20261109; 341 train / 114 validation / 114 held-out test. StandardScaler fit on genuine train only (values clipped to [-8,8]). Ten-latent-dimension conditional VAE, 35 maximum epochs, checkpoint chosen by validation reconstruction MSE; independent generator seeds 11,23,37. For each, generate 341 class-count-matched synthetic observations. Train logistic regression on 341 real observations alone, 341 real + 341 resampled real, or 341 real + 341 generated. Evaluate on identical 114 genuine held-out records; no test-based checkpoint selection.

| Training condition | Mean ROC AUC ± sample SD | Mean balanced accuracy ± sample SD |
|---|---:|---:|
| Real only | 0.99934 ± 0 | 0.99306 ± 0 |
| Matched real bootstrap | 0.99868 ± 0.00115 | 0.97553 ± 0.00974 |
| Real plus CVAE generated | 0.98071 ± 0.00884 | 0.94709 ± 0.02633 |

The synthetic arm **underperformed** both controls. The real-only classification scores are repeated across seeds because its data and deterministic training are unchanged; three CVAE seeds are NOT three independent patient cohorts. Generated clinical records have **not** passed privacy, calibration, fairness, out-of-site, clinical or medical usefulness validation. The CVAE samples have not been medically validated and are not for patient care.

```bash
python run.py --epochs 35
python test_outputs.py
```

Dependencies: NumPy, scikit-learn, PyTorch, Matplotlib. The Pass 26 chapter ZIP includes executed checkpoints, seed-wise 1026 held-out predictions, 1023 generated feature records, training traces, source hash/split manifest, and three code-generated plots. The source and checker are committed here; independent clean-checkout execution was not performed. **Full 16-unit Master QA remains HOLD; this single-modality experiment adds PARTIAL evidence to the multimodal precision-medicine application.**
