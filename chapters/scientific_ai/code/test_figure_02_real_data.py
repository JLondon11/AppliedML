"""Validation checks for authentic-data Scientific AI Figure 2.

Run: python chapters/scientific_ai/code/test_figure_02_real_data.py
"""
import json
import tempfile
from pathlib import Path
import numpy as np
from figure_02_scientific_discovery import main

def test_real_data_figure():
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)
        record=main(out)
        assert record["dataset_observations"]==309
        assert (record["training_windows"],record["validation_windows"],
                record["testing_windows"])==(178,60,59)
        assert record["dataset_years"]==[1700,2008]
        assert len(record["test_target_years"])==59
        assert len(record["observed_test"])==59
        assert len(record["ridge_predictions_test"])==59
        assert len(record["persistence_predictions_test"])==59
        assert all((out/f"figure_02_evolution_scientific_discovery.{ext}").is_file()
                   for ext in ("svg","png","pdf"))
        actual=np.array(record["observed_test"])
        pred=np.array(record["ridge_predictions_test"])
        persistence=np.array(record["persistence_predictions_test"])
        assert np.isclose(np.sqrt(np.mean((actual-pred)**2)),
                          record["heldout_rmse"],rtol=1e-12)
        assert np.isclose(np.sqrt(np.mean((actual-persistence)**2)),
                          record["persistence_heldout_rmse"],rtol=1e-12)
        assert all(np.isfinite(v) for v in pred)
        saved=json.loads((out/"figure_02_scientific_discovery_evidence.json").read_text())
        assert saved["dataset_values_float64_sha256"]==record["dataset_values_float64_sha256"]
        assert saved["selected_alpha"]==record["selected_alpha"]
        assert len(saved["alpha_validation_records"])==5

if __name__=="__main__":
    test_real_data_figure()
    print("PASS: genuine source data, chronological partition and output arithmetic verified")
