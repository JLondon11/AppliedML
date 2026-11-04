"""Real-data medical diagnosis case study for Chapter 1.

Dataset: Wisconsin Diagnostic Breast Cancer (UCI; distributed with scikit-learn).
Models: standardized logistic regression and random forest.
Outputs: metrics.csv, predictions.csv, ROC/confusion-matrix figure (SVG/PNG).
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1729

def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_breast_cancer(as_frame=True)
    X, y = ds.data, ds.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=SEED)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
    }

    rows, predictions = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(yte, pred),
            "balanced_accuracy": balanced_accuracy_score(yte, pred),
            "f1": f1_score(yte, pred),
            "roc_auc": roc_auc_score(yte, prob),
        })
        predictions[name] = (pred, prob)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    out = pd.DataFrame({"y_true": yte.to_numpy()}, index=yte.index)
    for name, (pred, prob) in predictions.items():
        out[f"{name}_pred"] = pred
        out[f"{name}_prob"] = prob
    out.to_csv(output_dir / "predictions.csv", index_label="row_id")

    best = metrics.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    pred, prob = predictions[best]
    cm = confusion_matrix(yte, pred)
    fpr, tpr, _ = roc_curve(yte, prob)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(fpr, tpr, label=f"{best} (AUC={roc_auc_score(yte, prob):.3f})")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "medical_diagnosis_roc.svg", bbox_inches="tight")
    fig.savefig(output_dir / "medical_diagnosis_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["malignant", "benign"])
    ax.set_yticks([0, 1], ["malignant", "benign"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "medical_diagnosis_confusion_matrix.svg", bbox_inches="tight")
    fig.savefig(output_dir / "medical_diagnosis_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "dataset": "Wisconsin Diagnostic Breast Cancer",
        "source": "UCI Machine Learning Repository via sklearn.datasets.load_breast_cancer",
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "split": "70/30 stratified holdout",
        "seed": SEED,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("results/medical_diagnosis_wdbc"))
    run(p.parse_args().output_dir)
