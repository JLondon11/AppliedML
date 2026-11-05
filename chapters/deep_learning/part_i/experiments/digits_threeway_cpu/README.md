# Chapter 2: Three-way train / validation / test CNN experiment

This CPU experiment uses the real **scikit-learn Digits dataset (1,797 labeled 8×8 grayscale images)**, **not MNIST**. Using stratification, we assigned **1,078 examples to training**, **359 to validation**, and **360 to held-out testing** (random seed **20261105**). There is no overlap between the partitions.

A 2-convolution-layer CNN (28,930 parameters) was trained with Adam (learning rate 0.002) for **51 epochs**. Only validation loss was used for early stopping and selection of the best checkpoint (**epoch 43**). After checkpoint selection, the model achieved **97.50% accuracy (351/360)**, **macro F1 = 0.97522**, and **cross-entropy = 0.09929** on the previously held-out testing split. This is one fixed split, not a multi-seed study or an MNIST benchmark.

The executable script `run.py` creates training and validation loss curves, a testing confusion matrix, prediction examples, convolutional feature maps, a model checkpoint and a JSON record of exact split indices. Run `python run.py` with numpy, scikit-learn, matplotlib and torch (CPU supported). All output files are regenerated in `outputs/`.

Dataset reference: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

The larger Chapter 2 MNIST, CIFAR-10, HAR, hurricane, Stanford Dogs, style transfer and optical-flow evaluations remain outstanding. Do not substitute these Digits findings for any of those experiments.
