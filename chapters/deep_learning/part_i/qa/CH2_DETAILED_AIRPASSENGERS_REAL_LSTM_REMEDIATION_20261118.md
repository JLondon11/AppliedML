# Detailed Chapter 2 — authentic airline LSTM science remediation (2026-11-18)

The **detailed smaller-font Chapter 2**, not the distinct short branch, was edited from the exact previous user-delivered source. Its "Application: Time Series Prediction with LSTM Using PyTorch" is now evaluated with the genuinely original historical AirPassengers monthly observations and the actually executed named LSTM.

## Source data, experimental validity and verification
Original `jbrownlee/Datasets/airline-passengers.csv`, blob SHA `3751f1359045b83e41406b3630686fb2639a448c`, reproduces **144 observed monthly counts** January 1949–December 1960. The source's record values were independently compared with the numerical reconstruction (original sum 40,363). The local normalized CSV differs in header quoting and line endings but has the same counts. A complete original-source trace, exact 144 records and local file SHA are packaged with the manuscript.

A **single-layer 16-hidden-unit PyTorch LSTM** using real 12-month historical windows was fitted with Adam, training-month-only normalization, **84 training target windows** (original months 13–96), **24 later validation windows**, and **24 untouched test months** (January 1959–December 1960). Minimum validation mean squared loss selected epoch **151**; the run ended at epoch **196** after 45 epochs without improvement. No test target entered checkpoint selection.

Two different protocols were measured: (1) one-month-ahead predictions using actual preceding observed history, and (2) autonomous **24-month recursive forecasts initialized at December 1958**, never ingesting held-out actual values. Matched original-data baseline predictions and error arrays are saved for both.

| Actual held-out experiment | LSTM RMSE (thousands) | Seasonal-naive RMSE (thousands) | Result |
|---|---:|---:|---|
| One-step with observed prior history | **75.9505** | **49.9867** | LSTM inferior |
| Truly recursive 24-month forecast | **101.2625** | **76.9946** | LSTM inferior |

One-step observed-history persistence RMSE **51.7820**; fully recursive constant-persistence RMSE **137.3290**. Each reported RMSE, MAE and bias was recomputed independently using its actual per-month saved predictions and original targets. Two complete local model runs generated **byte-identical numeric evidence** (SHA256 `1ebbe7c8c21d571e137392aab43ef36e64b91a018eff731fda79d229d573bca1`). The result supports a scientifically honest **scoped Level B offline historical forecasting experiment**, *not* superior airline-demand forecasting or a 2000-epoch 32.46-RMSE claim.

## Actual scientific and publication remediation
- Replaced `airline_training.png` placeholder with the actual four-panel `Figures_2_AirPassengers_RealData_LSTM_Forecast.pdf` saved in `figures/Figure_2_AirPassengers_RealData_LSTM_Forecast.pdf` under existing `fig:airline_loss` label (compiled **Figure 2.7**). Panel (a): actual original split time series; (b): actual one-step original test predictions; (c): true recursive 24-month forecast and matched seasonal baseline; (d): actual train/validation loss histories.
- Corrected the dataset description to **monthly international passenger counts**, removed unsupported claims of 2000 epochs, RMSE 32.46, and unverified training outcomes, and inserted truthful relative-error findings.
- Replaced an implicit full-data plotting code fragment with a short declared-input inference example; full actual runnable code, raw records, complete predictions and independent metric checks are saved in the user's ZIP at `code/real_airpassengers_lstm_application.py`, `evidence/original_airpassengers_lstm/` and committed GitHub analogue under `chapters/deep_learning/part_i/experiments/detailed_airpassengers_lstm/`.
- Compiled actual detailed LaTeX twice: **119 pages**, final PDF SHA256 `d36fe229edeb5b68cbdd4c349bf677cecc51aaefd7b77e0b1528ff4d529e84dc`. The scientific graphic is **embedded on PDF page 19** and visually inspected. No undefined citations/references or duplicate destination warnings.
- Preserved **73 figure environments, eight scientifically substantive technical tables, and 111 focused printed Python listings**. No blank/placeholder figure was promoted to authentic observed evidence.

## Master QA outcome

This pass raises **scoped scientific unit evidence from 2/12 to 3/12** in the current detailed edition. New scoped B-level evidence is **CH2-DETAIL-02**, the original AirPassengers LSTM application, for its exact 144-month offline task and observed negative results. Previously supported scopes remain: one original 8×8 Digits PyTorch CNN and one authentic NGDC sunspot initial-gradient mechanism. **Nine other named applications remain scientifically HOLD** (including UCI HAR, hurricane, 28×28 MNIST, CIFAR-10, Stanford Dogs, video style transfer). **34 fallback missing-image references and 11 overfull hbox events** remain in the actual 119-page PDF; **whole detailed chapter scientific and production approval HOLD**.

ZIP delivery: `Chapter2_Detailed_AirPassengers_RealData_Remediation_20261118.zip`, **169 archive members**, CRC verified. Includes complete current LaTeX, PDF, existing and new figure assets, source code, all original numeric records and outputs, matrix and QA report. The complete large manuscript and binary figure assets have not yet been verified committed into the canonical GitHub release directory. A GitHub Actions workflow has been authored but a passing run has not been verified.
