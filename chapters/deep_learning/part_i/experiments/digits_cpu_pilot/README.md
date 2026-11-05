# Chapter 2 — CPU real-data validation pilot

**This is NOT the MNIST benchmark.** The pilot uses the scikit-learn Digits dataset (1,797 examples, 8×8 grayscale). Its purpose is to verify the end-to-end CNN training and figure generation pipeline using an authentic dataset without external downloads.

Run `python run.py` with torch, numpy, scikit-learn, matplotlib.

Fixed split seed 17, stratified 75/25 train-test, 12 epochs, Adam learning rate 0.002. Model: Conv(1,12,3), Conv(12,24,3), max pooling, linear(384,48), linear(48,10).

Observed environment: PyTorch 2.10.0+cpu, Python 3.13.5.

One executed CPU run yielded: held-out accuracy 0.9422222222 and macro-F1 0.9412379617, 450 held-out images. See `metrics_verified.json`. Code regenerates the training loss figure, held-out predictions, confusion matrix, feature maps, model weights and complete metrics under `outputs/`.

This result must not be presented as MNIST, CIFAR-10, or as a published benchmark comparison. No claim of broad generalization is warranted by a single train/test split. Official dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

**Outstanding Chapter 2 evidence:** MNIST, CIFAR-10, UCI HAR/LSTM-GRU, NOAA/NRL/HURDAT2 hurricane images, Stanford Dogs transfer learning, optical-flow (KITTI/Sintel or actual video), style-transfer outputs and comparative ablations must be executed on their own datasets.

## Dataset and partition sizes

The real scikit-learn Digits dataset contains **1,797 labeled handwritten-digit images**, each of size **8 × 8 pixels**. A stratified random split (seed **17**) assigned **1,347 samples to training** and **450 to the held-out test set**. **No separate validation set was used (validation n = 0)**; thus the held-out accuracy and macro-F1 are test-set metrics from one split, not independent validation results. This dataset is **not MNIST**.
