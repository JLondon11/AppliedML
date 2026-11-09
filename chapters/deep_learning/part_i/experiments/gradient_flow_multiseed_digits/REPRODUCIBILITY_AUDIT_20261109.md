# Reproduction check — 2026-11-09

A fresh CPU execution of the archived `run.py` was compared against the archived `outputs/results.json` for the scikit-learn Digits experiment.

- Real dataset: 1,797 samples; training 1,078; validation 359; held-out test 360.
- Fixed split seed: 20261105; training seeds: 21, 42 and 84.
- Six runs: two architectures × three seeds; 35 epochs each.
- All six held-out accuracies, counts of correct test predictions, and validation-selected checkpoint epochs matched exactly.
- Maximum absolute change in test cross-entropy: **1.79e-7**.
- Maximum absolute change in best validation cross-entropy: **7.75e-7**.
- Maximum absolute change in initial layer gradient norm: **3.73e-9**.
- The tiny floating-point differences are consistent with platform-level numerical execution differences; metrics are reproducible to reported precision.

The generated vector figures `figures/heldout_accuracy_by_seed.svg`, `figures/initial_weight_gradient_norms.svg`, and `figures/mean_validation_crossentropy.svg` are data figures based on the archived experiment's measured outputs. The latter two SVGs use per-seed means rounded to 8 and 5 decimal places respectively for text-based vector plotting; full precision is available in `outputs/results.json` in the archival package.

**Scope limitation:** This is an auxiliary gradient-flow investigation on scikit-learn Digits. It does not certify the Chapter 2 named MNIST, HAR, CIFAR-10, hurricane, dog-breed, style-transfer, optical-flow, or recurrent LSTM/GRU studies. The official MNIST host could not be reached from the CPU environment during this pass.
