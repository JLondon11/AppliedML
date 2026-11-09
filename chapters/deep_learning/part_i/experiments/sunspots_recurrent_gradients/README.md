# Chapter 2 — Recurrent vanishing-gradient sensitivity from real sunspot data

This is an executable empirical gradient-analysis experiment using **309 annual sunspot observations (1700–2008)** from the National Geophysical Data Center, distributed through statsmodels. The experiment compares initialized vanilla RNN, LSTM, and GRU models, each with 16 hidden units and a regression head, on real chronological 12-year and 36-year training windows. It measures squared-loss input-gradient magnitudes by PyTorch automatic differentiation, rather than inventing a synthetic sequence.

For 12-year windows, there are **178 training, 60 validation, and 59 testing windows**. For 36-year windows, the counts are **154 training, 60 validation, and 59 testing windows**. Twenty-four actual training sequences are used for the gradient calculations at each length and seed; no validation or test sequences enter the gradient calculation. Seeds: 17, 42, and 84.

The median oldest-to-newest input-gradient ratios at 36 years were RNN **2.77e-9**, LSTM **4.76e-8**, and GRU **3.02e-8**. They show severe gradient attenuation for all three randomly initialized models, with a larger median ratio for the gated architectures in this narrowly specified comparison.

Run `python run.py` with Python, numpy, torch, matplotlib and statsmodels. It produces a complete `outputs/results.json` containing all gradient vectors, run metadata, source hash, and three figures in PNG/PDF formats. The full figures and measured result JSON are also included in the downloadable archival package. A GitHub vector figure shows measured 36-year gradient ratios.

**Scope:** These networks were not trained in this sensitivity experiment; no forecasting accuracy is claimed. The separately trained `sunspots_lstm_gru` case study supplies forecasting predictions. This study does not certify the missing MNIST, CIFAR-10, UCI HAR, hurricane, dog-breed, neural style transfer or optical flow experiments.

References: https://www.statsmodels.org/stable/datasets/generated/sunspots.html ; Hochreiter & Schmidhuber (1997), https://doi.org/10.1162/neco.1997.9.8.1735 ; Cho et al. (2014), https://arxiv.org/abs/1406.1078 .
