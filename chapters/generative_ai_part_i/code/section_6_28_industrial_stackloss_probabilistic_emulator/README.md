# Chapter 6 §6.28 — Industrial process conditional generator (Pass 36)

**Genuine data:** Brownlee's historical ammonia oxidation plant, 21 measured daily operating records, distributed as `statsmodels.datasets.stackloss`. Source: Brownlee (1965), *Statistical Theory and Methodology in Science and Engineering*, second edition, pp. 491–500; [statsmodels dataset documentation](https://www.statsmodels.org/stable/datasets/generated/stackloss.html).

**Purpose:** A tractable real-data industrial *process emulator*, not a closed-loop physical digital twin. Inputs are AIRFLOW, WATERTEMP and ACIDCONC; response is measured STACKLOSS (10 × percentage of input ammonia escaping the absorption column).

**Models:** Conditional Gaussian process (Matérn 3/2 plus fitted WhiteKernel), ridge regression, training-mean reference. Leave-one-observation-out cross-validation, with training-only feature standardization and all hyperparameters fitted inside each fold. 21 held-out measurements per model; 100 GP predictive draws for each held-out condition (2,100 generated numbers). Metrics use **only genuine measured plant outputs**.

| Arm | LOO RMSE | LOO MAE |
|---|---:|---:|
| Gaussian-process generator | 4.11893 | 2.96667 |
| Ridge regression | **3.64766** | **2.93444** |
| Training-mean reference | 10.42281 | 7.88095 |

Empirical coverage of nominal central 90% GP interval: **19/21 = 90.48%**; mean interval width: **10.0970**. The generator produces **2 negative out of 2,100 draws**, violating the physical nonnegativity constraint. This violation remains visible in the research report. It is not evidence that the plant had negative stack loss.

**Run:** `python run.py` then `python test_outputs.py`. Dependencies: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib. Generates full out-of-fold measurements and predictions, all posterior samples, SHA-256 source manifest, scientific vector PDF and PNG figures. The complete executed dataset-derived evidence is in the Chapter 6 Pass 36 archive.

**Certification:** PARTIAL real-data evidence for the industrial-digital-twin application, with a documented negative comparison. 21 observations cannot establish independent plant generalization, physics compliance, sensor-stream dynamics, fault diagnosis, or closed-loop control. Full case-study certification remains HOLD.
