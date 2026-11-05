# Chapter 2 — Multi-seed gradient-flow comparative case study

The experiment uses **1,797 real grayscale 8 × 8 handwritten digits** from scikit-learn Digits (**not MNIST**). A fixed stratified split (seed 20261105) assigned **1,078 examples to training, 359 to validation, and 360 to held-out testing**. Three initialization seeds (21, 42, 84) were evaluated for each architecture, using identical partitions and 35 epochs per run.

Two equal-parameter architectures (38,090 weights each) were compared: a plain eight-layer tanh MLP and an eight-layer MLP with scaled residual connections (scale 0.25). Both used Adam with learning rate 0.001. For every architecture and seed, **minimum validation cross-entropy** determined the checkpoint. Testing was evaluated only after selecting the checkpoint.

Actual CPU test accuracy for the plain network: 82.22%, 90.83%, 82.78% across seeds 21,42,84 (mean 85.28%, sample SD 4.82 percentage points). The residual network achieved 96.39%, 95.83%, 96.11% (mean 96.11%, sample SD 0.28 percentage points). Validation loss and initial weight-gradient L2 norms are recorded in the machine-readable execution package.

Run `python run.py` using Python with torch, numpy, scikit-learn, matplotlib. The script regenerates `outputs/results.json` and PDF/PNG charts for test accuracy, validation cross-entropy, and initialization gradient magnitudes.

This is a three-seed, one-split experiment, not a published benchmark or MNIST result. Results are specific to these architectures, optimizer settings, and dataset. Dataset reference: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
