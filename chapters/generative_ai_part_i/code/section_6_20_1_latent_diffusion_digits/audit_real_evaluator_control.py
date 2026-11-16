"""Independent real-data evaluator control for Chapter 6 latent diffusion digits.

Re-execute the exact original train/validation/test split and logistic digit
classifier from run.py. Generated class counts are previously saved model outputs:
this audit does not retrain or resample the diffusion generator.

python audit_real_evaluator_control.py --counts figure_6_44_class_counts_pass27.csv --out original_evaluator_control
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261109


def run(counts: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    raw = counts.read_bytes()
    with counts.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    expected = {(arm, label) for arm in ("conditional", "unconditional")
                for label in ("overall", *(str(i) for i in range(10)))}
    if len(saved) != 22 or {(r["arm"], r["label"]) for r in saved} != expected:
        raise ValueError("Expected 22 unique, original per-class model-output rows")
    source = load_digits()
    x = (source.data / 16).astype("float32")
    y = source.target.astype("int64")
    ids = np.arange(len(y))
    train, other = train_test_split(ids, test_size=.3, random_state=SEED, stratify=y)
    val, test = train_test_split(other, test_size=.5, random_state=SEED, stratify=y[other])
    assert (len(train), len(val), len(test)) == (1257, 270, 270)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    model = LogisticRegression(max_iter=2000, random_state=SEED).fit(x[train], y[train])
    cm = confusion_matrix(y[test], model.predict(x[test]), labels=range(10))
    saved = {(r["arm"], r["label"]): r for r in saved}
    for arm in ("conditional", "unconditional"):
        if any(int(saved[arm, str(i)]["n"]) != 100 for i in range(10)):
            raise ValueError("Model-output class count is not 100")
        if sum(int(saved[arm, str(i)]["correct"]) for i in range(10)) != int(saved[arm, "overall"]["correct"]):
            raise ValueError("Model-output overall does not reconcile with class rows")
    records = []
    for i in range(10):
        n = int((y[test] == i).sum())
        correct = int(cm[i, i])
        con = int(saved["conditional", str(i)]["correct"])
        unc = int(saved["unconditional", str(i)]["correct"])
        records.append(dict(digit=i, real_test_n=n, real_test_correct=correct,
                            real_test_recall=correct/n, conditional_generated_n=100,
                            conditional_generated_agree=con, conditional_generated_agreement=con/100,
                            unconditional_generated_n=100, unconditional_generated_agree=unc,
                            unconditional_generated_agreement=unc/100))
    with (out / "real_evaluator_vs_generated_class.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (out / "real_evaluator_confusion.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["observed\\predicted"] + list(range(10)))
        for i, row in enumerate(cm):
            writer.writerow([i] + list(map(int, row)))
    report = {
        "original_dataset": "sklearn.datasets.load_digits original 1797 handwritten digit scans",
        "original_source_n": 1797, "seed": SEED,
        "train_n": len(train), "validation_n": len(val), "test_n": len(test),
        "original_test_index_sha256": hashlib.sha256(test.astype("<i8").tobytes()).hexdigest(),
        "generated_class_counts_sha256": hashlib.sha256(raw).hexdigest(),
        "evaluator": "LogisticRegression(max_iter=2000,random_state=20261109), original training images only",
        "real_evaluator_correct": int(np.trace(cm)), "real_evaluator_test_n": len(test),
        "real_evaluator_accuracy": float(np.trace(cm)/len(test)),
        "recorded_conditional_agreement": int(saved["conditional", "overall"]["correct"])/1000,
        "recorded_unconditional_comparison_agreement": int(saved["unconditional", "overall"]["correct"])/1000,
        "n_original_independent_real_test_scans": len(test),
        "n_recorded_generated_samples_per_arm": 1000,
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "limitations": [
            "Generated class values are previously saved model output; diffusion model not retrained",
            "Real test and generated images are distinct distributions; descriptive, not paired",
            "Generated images may be correlated; no valid generator-seed uncertainty estimated",
            "Classifier agreement is not a human-rated image realism metric",
            "Unconditional comparison class labels were assigned, not prompted"
        ]
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6),
                             gridspec_kw={"width_ratios": [1.12, 1]},
                             layout="constrained")
    fig.suptitle("Original-digit classifier control for conditional generation",
                 fontsize=15, fontweight="semibold")
    ticks = np.arange(10)
    real = np.array([r["real_test_recall"] for r in records])
    generated = np.array([r["conditional_generated_agreement"] for r in records])
    for i in ticks:
        axes[0].plot([i, i], [generated[i], real[i]], color="#adbdc5", lw=1.7)
    axes[0].scatter(ticks-.08, real, color="#21677a", s=52,
                    label="Original held-out recall", zorder=3)
    axes[0].scatter(ticks+.08, generated, color="#b26048", s=52,
                    label="Generated conditional agreement", zorder=3)
    axes[0].set(xlabel="Digit class", ylabel="Class-specific correct fraction",
                ylim=(0, 1.08), xticks=ticks,
                title="A  Evaluator on real vs generated inputs")
    axes[0].grid(axis="y", alpha=.18)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8)
    image = axes[1].imshow(cm, cmap="Blues", vmin=0, vmax=30, interpolation="nearest")
    axes[1].set(xticks=ticks, yticks=ticks, xlabel="Predicted digit on real test scan",
                ylabel="Observed digit", title="B  Real held-out evaluator confusion")
    for i in ticks:
        for j in ticks:
            if cm[i, j]:
                axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                             fontsize=8, color="white" if cm[i, j] >= 18 else "#25485e")
    fig.colorbar(image, ax=axes[1], shrink=.83, label="Original observed test images")
    fig.savefig(out / "real_classifier_control.png", dpi=210)
    fig.savefig(out / "real_classifier_control.svg")
    plt.close(fig)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.counts, a.out), indent=2))
