# Chapter 2 — Real-data LSTM versus GRU time-series forecasting

The experiment uses **309 genuine annual sunspot observations from 1700–2008**, distributed by `statsmodels.datasets.sunspots` and originally sourced from the National Geophysical Data Center. A 12-year history forecasts the immediately following year's observed sunspot number, producing **297 supervised windows**. A chronology-based split uses **178 training windows (target years 1712–1889)**, **60 validation windows (1890–1949)** and **59 held-out testing windows (1950–2008)**. Train-only statistics normalize the data.

Three seeds (17, 42 and 84) each train an LSTM and a GRU using 16 hidden units and a linear regression head. Both use Adam (learning rate 0.006), batch size 32, 1.0 gradient clipping, at most 90 epochs and validation-only checkpoint selection with early stopping patience 15.

Measured held-out test RMSE: LSTM **20.6796 mean (0.4830 sample SD)** and GRU **20.6677 mean (2.1251 sample SD)** sunspots. Persistence (copy previous annual value) RMSE: **33.1750**. The LSTM and GRU means are virtually identical, with notable GRU seed sensitivity. No general architecture ranking follows from three seeds and one split.

Execute `python run.py`; this script generates `outputs/results.json`, all numerical metrics and predictions, and three scientific PNG/PDF figures: held-out observed-versus-predicted sunspot series, RMSE by seed (including persistence), and validation-loss histories. A complete executed run with authentic observations and those figures is preserved in the downloadable archival package.

**Protocol qualification:** This is a one-step-ahead forecast using the previous 12 *observed* annual values; no future target is used as an input. It is not a multistep recursive forecast. This satisfies real-data evidence for a new sunspot-forecasting comparison, but does not replace missing results for HAR, MNIST, CIFAR-10, hurricane, Stanford Dogs, style transfer or optical flow.

Data source: https://www.statsmodels.org/stable/datasets/generated/sunspots.html (NGDC). LSTM citation: Hochreiter and Schmidhuber (1997), https://doi.org/10.1162/neco.1997.9.8.1735 . GRU citation: Cho et al. (2014), https://arxiv.org/abs/1406.1078 .
