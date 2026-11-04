"""Real-data edge-ML quantization case study for Chapter 1.

Dataset: scikit-learn handwritten digits.
Model: multinomial logistic regression.
Experiment: compare full-precision coefficients with symmetric int8 quantized weights.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 1729

def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(
        ds.data, ds.target, test_size=0.30, random_state=SEED, stratify=ds.target
    )
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = LogisticRegression(
        max_iter=5000, solver="lbfgs", random_state=SEED
    )
    model.fit(Xtr_s, ytr)
    full_pred = model.predict(Xte_s)

    W = model.coef_.astype(np.float64)
    b = model.intercept_.astype(np.float64)
    scale = np.max(np.abs(W)) / 127.0
    Wq = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    logits_q = Xte_s @ (Wq.astype(np.float64) * scale).T + b
    quant_pred = np.argmax(logits_q, axis=1)

    metrics = pd.DataFrame([
        {
            "representation": "float64",
            "accuracy": accuracy_score(yte, full_pred),
            "weight_bytes": W.nbytes,
        },
        {
            "representation": "int8",
            "accuracy": accuracy_score(yte, quant_pred),
            "weight_bytes": Wq.nbytes,
        },
    ])
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame({
        "y_true": yte,
        "float64_prediction": full_pred,
        "int8_prediction": quant_pred,
    }).to_csv(output_dir / "predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    ax.bar(metrics["representation"], metrics["weight_bytes"] / 1024)
    ax.set_ylabel("Coefficient storage (KiB)")
    fig.tight_layout()
    fig.savefig(output_dir / "edge_weight_storage.svg", bbox_inches="tight")
    fig.savefig(output_dir / "edge_weight_storage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    ax.bar(metrics["representation"], metrics["accuracy"])
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Held-out accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "edge_accuracy.svg", bbox_inches="tight")
    fig.savefig(output_dir / "edge_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(yte, quant_pred)
    np.savetxt(output_dir / "int8_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    (output_dir / "provenance.json").write_text(json.dumps({
        "dataset": "Optical Recognition of Handwritten Digits",
        "source": "UCI-derived sklearn.datasets.load_digits",
        "split": "70/30 stratified holdout",
        "quantization": "per-tensor symmetric int8 coefficients",
        "seed": SEED,
    }, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("results/edge_digits_quantization"))
    run(p.parse_args().output_dir)
