# Section 6.13.1 — Real Mauna Loa atmospheric CO₂ conditional VAE

This is an **executed observational sequence-generation diagnostic**, not a numerical Earth System Model or weather-model emulation. The model taught in Chapter 6 is a **conditional VAE**. Data come from the real Mauna Loa weekly atmospheric CO₂ dataset bundled in `statsmodels.datasets.co2` (1958–2001). Documentation: https://www.statsmodels.org/stable/datasets/generated/co2.html.

### Run
Install `torch`, `numpy`, `pandas`, `scikit-learn`, `statsmodels`, and `matplotlib`. From the repository root:

```bash
python chapters/generative_ai_part_i/code/section_6_13_1_climate_co2_cvae/run.py --epochs 65
python chapters/generative_ai_part_i/code/section_6_13_1_climate_co2_cvae/test_outputs.py
```

Produces three CVAE checkpoints, test predictions for every seed and every 8-week forecast horizon, validation/training traces, an input-data SHA-256 manifest, numerical metrics and two result figures.

### Observed split and evaluation

52-week historical conditioning window, eight-week forecast. Strict chronological train/validation/test partitions: complete forecast targets in training through 1990, in validation 1991–1995, and test forecast starts from 1996 to 2001. There are 1,145 train, 254 validation, and 306 held-out forecast windows. Windows containing missing values are dropped instead of future-aware interpolation. Validation mean-squared error selects the checkpoint; all test comparisons are made using the **prior predictive mean**, not the posterior encoder that observes the true future.

### Measured test results (three seeds; units ppm)

| Method | RMSE | MAE |
|---|---:|---:|
| Conditional VAE predictive mean | 0.4868 | 0.3856 |
| Ridge baseline | 0.6673 | 0.5244 |
| Last-value persistence | 1.4799 | 1.1958 |
| 52-week seasonal drift | 2.0913 | 1.8574 |

**Important negative uncertainty result:** The nominal central 90% prior-predictive interval covers only 20.25% of test targets across seeds. Thus the model does **not** provide calibrated probabilistic forecasts. Three seeds are not three independent atmospheric records, and the rolling forecast windows overlap extensively.

This limited demonstration is classified as **PARTIAL evidence** in the chapter's 16-unit matrix, with publication certification still **HOLD**.

Locally generated figures, per-window predictions, checkpoints, and source trace are supplied in the Pass 19 chapter evidence ZIP. GitHub commits include the full runnable generator and the numerical summary; binary checkpoints and figures are retained in the evidence ZIP rather than in this GitHub source folder.
