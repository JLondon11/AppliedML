# Chapter 2 — gradient flow on real Digits data

This case study compares plain tanh and scaled residual tanh networks using the **scikit-learn Digits** dataset (8×8 real handwritten digits). This is **not** an MNIST, CIFAR-10, or HAR benchmark. The same 1,347/450 stratified split (seed 2705), architecture depth/width and parameter count (38,090) were used, with 20 training epochs and Adam (lr 0.001).

One CPU run measured held-out accuracy of **85.33%** for the plain network and **96.89%** for the residual network; macro F1 was 0.8558 and 0.9689 respectively. Initial layer-wise gradient magnitudes were more uniform for the residual model, as measured before any parameter updates. Because initialization paths and model structure differ, this is not evidence of a universal superiority claim; repeated seeds, other architectures, and datasets are required.

Run: `python run.py` with torch, numpy, scikit-learn, matplotlib. The script regenerates metrics.json, held-out accuracy/loss curves, gradient norm plot and both confusion matrices inside `outputs/`.

Dataset citation: [scikit-learn `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Training protocol and measured metrics are saved in `metrics_verified.json`. The generated graphics are available in the companion artifact package; GitHub figure-binary publication is pending.

## Dataset and partition sizes

The real scikit-learn Digits dataset contains **1,797 labeled handwritten-digit images**, each of size **8 × 8 pixels**. A stratified random split (seed **2705**) allocated **1,347 images to training** and **450 images to held-out testing**, used identically for the plain and residual networks. **No separate validation subset was used (validation n = 0)**. Training losses were computed on the training subset; reported final accuracies and macro-F1 scores come from the held-out test subset. This experiment is **not MNIST**.
