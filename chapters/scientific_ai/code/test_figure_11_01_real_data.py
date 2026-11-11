"""Recompute the Scientific AI Figure 11.1 outputs from genuine NGDC sunspots."""
import tempfile
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from figure_11_01_observed_sunspots import experiment

def test_observed_source_and_predictions():
    with tempfile.TemporaryDirectory() as tmp:
        result=experiment(Path(tmp))
        assert result["original_records"]==309
        assert result["original_years"]==[1700,2008]
        assert (result["train_windows"],result["validation_windows"],
                result["test_windows"])==(178,60,59)
        assert len(result["test_actual"])==59
        assert result["chosen_alpha"] in [0.1,1,10,100,1000]
        y=np.asarray(result["test_actual"])
        p=np.asarray(result["test_predicted"])
        q=np.asarray(result["test_persistence"])
        assert np.isclose(np.sqrt(mean_squared_error(y,p)),result["test_rmse"],rtol=1e-12)
        assert np.isclose(np.sqrt(mean_squared_error(y,q)),result["persistence_rmse"],rtol=1e-12)
        for ext in ("png","pdf","svg"):
            assert (Path(tmp)/f"Figure_11_01_Scientific_AI_Ecosystem.{ext}").is_file()
        assert (Path(tmp)/"Figure_11_01_measured_evidence.json").is_file()

if __name__=="__main__":
    test_observed_source_and_predictions()
    print("PASS: genuine sunspots, correct splits, numerically matched forecasts, generated figures")
