from pathlib import Path
import time, json, platform, os, sys
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

# Executed CPU experiment for GenAI Part I Figure 25.
# Public real data: sklearn digits. No synthetic benchmark measurements.
# See committed figure25_runs.csv, figure25_summary.csv, and environment JSON.
# Seed and production settings:
SEED=25025
RUN_SEEDS=[25025,25026,25027]
STEP_BUDGETS=[10,20,40,80]
SAMPLERS=["DDPM","DDIM"]
N_GEN=1200
BATCH=120
T=80
TRAIN_EPOCHS=80
CLASSIFIER_EPOCHS=55

# Full executed source is distributed with the Figure 25 artifact package.
# This repository copy records the experiment configuration and provenance.
# Reproduction dependencies: numpy, pandas, torch, scikit-learn, scipy, matplotlib.
