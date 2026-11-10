# Chapter 6 — Concrete-Mix Generation and Retrospective Strength Screening

**Real measured source:** UCI Concrete Compressive Strength Dataset (Yeh), 1,030 observations with seven ingredient quantities (kg/m³), curing age (days) and measured compressive strength (MPa). Source https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength . Initial mirror https://github.com/Ankit-RV/Concrete-compressive-strength/blob/main/Concrete_Data_Yeh.csv, original Git blob SHA 93f829fc160bac648bc84d034574a18b3685a2ef. Primary publication: I.-C. Yeh, Cement and Concrete Research 28(12), 1797–1808 (1998).

**Execution and source:** Node 18 or later: `node run.mjs Concrete_Data_Yeh.csv results.json`; figure regeneration requires NumPy, pandas and Matplotlib: `python publication_figures.py`.

**Leakage control:** All samples with identical seven-ingredient formulations are grouped even if curing age differs, giving 427 unique exact mix groups. Deterministic group partitions: 277 groups / 645 training samples; 64 groups / 156 validation; 86 groups / 229 held-out test. Validation chooses k among 3, 5, 9 and 15; five-neighbor standardized and log-age-transformed inverse-distance regression is selected.

| Prediction method | Genuine held-out RMSE, MPa | Genuine held-out MAE, MPa |
|---|---:|---:|
| Validated five-neighbor surrogate | **7.416** | **5.288** |
| Training-mean baseline | 16.700 | 13.678 |

**Generative task:** A conditional empirical Gaussian-kernel model samples ingredient vectors near authentic 28-day training mixtures whose **measured** strength exceeded 40 MPa. The unconditional arm samples near all 28-day training mixtures. Each arm generates 300 **new numerical proposals**. Simplified rule screen: water/binder 0.25–0.65, binder 200–650 kg/m³, constituent sum 1500–2700 kg/m³ and ingredient nonnegativity. The generator produced 296/300 conditional and 280/300 unconditional rule-feasible formulas; 267 and 91 of those scored ≥40 MPa **according to the surrogate only**.

No generated mixture was cast or physically tested; strength predictions for generated candidates are NOT genuine laboratory observations. On an exploratory stricter low-cement rule (cement ≤251.8 kg/m³), 29 held-out real 28-day mixtures passed the composition screen and **none** had a measured strength ≥40 MPa. These negative findings must be reported; no verified high-strength low-cement candidate is established.

**Evidence:** Full real-data CSV, exact generative/validation implementation, benchmark_summary.json, all 229 held-out real-strength predictions, 200 saved generated candidate records, and two publication figure scripts/assets. The deterministic local output file results.json also contains all 600 generated mix candidates. No experiment has validated new mixes, structural safety, workability, or durability.

**Case-study scope:** Real-data probabilistic concrete-mix candidate proposal and retrospective strength screening, not physical autonomous engineering design. Scientific publication certification **HOLD** pending experimental validation of the generated mixes.