"""Validate Figure 8 genuine WDBC model outputs and held-out predictions.

Usage: python chapters/scientific_ai/code/test_figure_08_real_data.py
Generates evidence by executing the neighboring experiment script.
"""
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

SOURCE=Path(__file__).with_name("figure_08_wdbc_baselines.py")

def test_wdbc_experiment():
    with tempfile.TemporaryDirectory() as d:
        # Source writes to its own outputs directory by design.
        subprocess.run([sys.executable,str(SOURCE)],check=True,cwd=SOURCE.parent)
        root=SOURCE.parent/"outputs"
        record=json.loads((root/"results.json").read_text())
        assert record["observations"]==569
        assert record["feature_count"]==30
        assert record["training"]==455
        assert record["heldout_testing"]==114
        assert record["cross_validation_folds"]==5
        assert len(record["fold_results"])==20
        assert len(record["heldout_metrics"])==4
        with (root/"heldout_predictions.csv").open(newline="") as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==4*114
        for outcome in record["heldout_metrics"]:
            group=[x for x in rows if x["model"]==outcome["model"]]
            labels=np.asarray([int(x["true_label"]) for x in group])
            pred=np.asarray([int(x["predicted_label"]) for x in group])
            prob=np.asarray([float(x["probability_benign"]) for x in group])
            assert len(group)==114
            assert math.isclose(accuracy_score(labels,pred),outcome["test_accuracy"],rel_tol=1e-12)
            assert math.isclose(roc_auc_score(labels,prob),outcome["test_auroc"],rel_tol=1e-12)
        for ext in ("png","pdf","svg"):
            assert (root/f"figure_08_wdbc_classical_baselines.{ext}").stat().st_size>1000

if __name__=="__main__":
    test_wdbc_experiment()
    print("PASS: genuine WDBC source, 20 CV folds, 456 test predictions, 4 ROC scores and 3 figure formats")
