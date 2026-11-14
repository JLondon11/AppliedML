"""Fail-closed verification of saved original Wisconsin cytology CVAE records.

Run this after the committed run.py has executed:
python verify_pass54_original_data.py results

Source model training remains in run.py. Passing this check does not certify
clinical use, cross-hospital generalization or patient privacy.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

EXPECTED = {
    "real_only": (0.9993386243386243, 0.9930555555555557),
    "real_bootstrap": (0.9986772486772487, 0.9755291005291006),
    "real_plus_cvae": (0.98070987654321, 0.9470899470899471),
}

def audit(root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    d = load_breast_cancer()
    x = d.data.astype(np.float32)
    y = d.target.astype(np.int64)
    digest = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    # Accept both canonical committed-run and independent-replay manifest keys.
    if digest != m.get("data_sha256", m.get("sha256")):
        raise AssertionError("Real WDBC source fingerprint mismatch")
    counts = (m.get("train", m.get("training")), m.get("validation"),
              m.get("test", m.get("testing")))
    if counts != (341, 114, 114):
        raise AssertionError("Wrong original-data split counts")
    split = m.get("splits")
    splits = ([split["train_ids"], split["validation_ids"], split["test_ids"]] if split
              else [m["split_train"], m["split_validation"], m["split_test"]])
    if sorted(sum(splits, [])) != list(range(569)):
        raise AssertionError("Split leakage or omitted original subjects")
    with (root / "metrics.csv").open(newline="") as f:
        metrics = list(csv.DictReader(f))
    with (root / "test_predictions.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (root / "training.csv").open(newline="") as f:
        history = list(csv.DictReader(f))
    with (root / "prior_samples.csv").open(newline="") as f:
        samples = list(csv.DictReader(f))
    if tuple(map(len, (metrics, rows, history, samples))) != (9, 1026, 105, 1023):
        raise AssertionError("Evidence record count incorrect")
    for arm, (target_auc, target_balanced_accuracy) in EXPECTED.items():
        measured = []
        for seed in (11, 23, 37):
            sub = [r for r in rows if int(r["seed"]) == seed and r["arm"] == arm]
            if len(sub) != 114 or sorted(int(r["test_index"]) for r in sub) != list(range(114)):
                raise AssertionError("Repeated or missing original test subjects")
            true = np.array([int(r["true_label"]) for r in sub])
            probs = np.array([float(r["probability_benign"]) for r in sub])
            predicted = np.array([int(r["prediction"]) for r in sub])
            if not np.array_equal(true, y[np.array(splits[2])]):
                raise AssertionError("Misidentified original held-out label")
            if not np.array_equal(predicted, (probs >= 0.5).astype(int)):
                raise AssertionError("Recorded classification conflicts with probabilities")
            auc = roc_auc_score(true, probs)
            ba = balanced_accuracy_score(true, predicted)
            saved = next(v for v in metrics if int(v["seed"]) == seed and v["arm"] == arm)
            if abs(float(saved["ROC_AUC"]) - auc) > 1e-12:
                raise AssertionError("Per-seed AUC mismatch")
            if abs(float(saved["balanced_accuracy"]) - ba) > 1e-12:
                raise AssertionError("Per-seed balanced accuracy mismatch")
            measured.append((auc, ba))
        if abs(np.mean([a for a, _ in measured]) - target_auc) > 1e-10:
            raise AssertionError("Mean ROC AUC mismatch")
        if abs(np.mean([b for _, b in measured]) - target_balanced_accuracy) > 1e-10:
            raise AssertionError("Mean balanced accuracy mismatch")
    for seed in (11, 23, 37):
        if sum(int(r["seed"]) == seed for r in samples) != 341:
            raise AssertionError("Generated feature record count incorrect")
        if sum(int(r["seed"]) == seed for r in history) != 35:
            raise AssertionError("Training trace incomplete")
        if (root / f"cvae_seed{seed}.pt").stat().st_size < 10000:
            raise AssertionError("Trained checkpoint absent")
    # The committed runner emits scientific plots under figures/.
    figure_folder = root / "figures"
    if not ((figure_folder / "real_patient_auc_ablation.png").is_file()
            or (root / "wdbc_original_test_ablation.png").is_file()):
        raise AssertionError("Missing actual test-result figure")
    return dict(status="PASS_SCOPED_REAL_DATA_AUDIT",source_cases=569,
                training=341, validation=114, heldout=114,
                saved_prediction_rows=1026, saved_generated_rows=1023,
                seeds=3, clinical_validation="NOT PERFORMED")

if __name__ == "__main__":
    import sys
    print(json.dumps(audit(sys.argv[1] if len(sys.argv) > 1 else
                           Path(__file__).parent / "results"), indent=2))
