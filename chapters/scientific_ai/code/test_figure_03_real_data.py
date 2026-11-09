"""Regression checks for real UCI WDBC Scientific AI Figure 3.
Usage: python chapters/scientific_ai/code/test_figure_03_real_data.py
"""
import json
import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from figure_03_scientific_ai_evolution import main

def test_real_observation_figures():
    with tempfile.TemporaryDirectory() as temp:
        result=main(Path(temp))
        assert (result["n_samples"],result["n_features"])==(569,30)
        assert (result["training"],result["validation"],result["testing"])==(341,114,114)
        assert len(result["test_labels"])==114
        assert len(result["test_predicted_labels"])==114
        assert len(result["test_probabilities_malignant"])==114
        y=np.asarray(result["test_labels"])
        pred=np.asarray(result["test_predicted_labels"])
        probs=np.asarray(result["test_probabilities_malignant"])
        assert np.array_equal(confusion_matrix(y,pred,labels=[0,1]),
                              np.asarray(result["test_confusion_matrix_true_benign_malignant"]))
        assert np.isclose(accuracy_score(y,pred),result["test_accuracy"])
        assert np.isclose(roc_auc_score(y,probs),result["test_roc_auc"])
        assert all((Path(temp)/f"figure_03_evolution_scientific_ai.{ext}").is_file()
                   for ext in ("png","svg","pdf"))
        record=json.loads((Path(temp)/"figure_03_scientific_ai_real_data_evidence.json").read_text())
        assert record["source_data_float64_sha256"]==result["source_data_float64_sha256"]

if __name__=="__main__":
    test_real_observation_figures()
    print("PASS: observed records, disjoint partitions, output arithmetic and figures")
