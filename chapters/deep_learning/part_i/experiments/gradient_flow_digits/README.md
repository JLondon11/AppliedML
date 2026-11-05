# Chapter 2 — gradient flow on real Digits data

This case study compares plain tanh and scaled residual tanh networks using the **scikit-learn Digits** dataset (8×8 real handwritten digits). This is **not** an MNIST, CIFAR-10, or HAR benchmark. The same 1,347/450 stratified split (seed 2705), architecture depth/width and parameter count (38,090) were used, with 20 training epochs and Adam (lr 0.001).

One CPU run measured held-out accuracy of **85.33%** for the plain network and **96.89%** for the residual network; macro F1 was 0.8558 and 0.9689 respectively. Initial layer-wise gradient magnitudes were more uniform for the residual model, as measured before any parameter updates. Because initialization paths and model structure differ, this is not evidence of a universal superiority claim; repeated seeds, other architectures, and datasets are required.

Run: `python run.py` with torch, numpy, scikit-learn, matplotlib. The script regenerates metrics.json, held-out accuracy/loss curves, gradient norm plot and both confusion matrices inside `outputs/`.

Dataset citation: [scikit-learn `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Training protocol and measured metrics are saved in `metrics_verified.json`. The generated graphics are available in the companion artifact package; GitHub figure-binary publication is pending.
