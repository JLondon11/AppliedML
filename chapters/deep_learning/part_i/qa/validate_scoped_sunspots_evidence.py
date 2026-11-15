"""Recompute real-dataset, held-out forecast and numeric-scoring checks for Chapter 2.

Usage: python validate_scoped_sunspots_evidence.py path/to/results.json
The full experiment generator remains in ../experiments/sunspots_lstm_gru/run.py.
"""
import argparse, hashlib, json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.datasets import sunspots

def verify(filename):
    results=json.load(open(filename))
    raw=sunspots.load_pandas().data
    original=raw.SUNACTIVITY.to_numpy(dtype=np.float32)
    assert len(original)==309
    assert hashlib.sha256(original.tobytes()).hexdigest()==results["source_values_sha256_float32"]
    assert (results["training_windows"],results["validation_windows"],results["testing_windows"])==(178,60,59)
    assert results["test_years"]==list(range(1950,2009))
    truth=np.array(results["ground_truth_test"])
    assert truth.shape==(59,) and np.allclose(truth,original[250:],atol=1e-5)
    assert len(results["results"])==6
    for record in results["results"]:
        key=record["architecture"]+"_"+str(record["seed"])
        p=np.array(results["forecasts"][key])
        assert p.shape==(59,)
        assert abs(float(np.sqrt(mean_squared_error(truth,p)))-record["test"]["rmse"])<1e-4
        assert abs(float(mean_absolute_error(truth,p))-record["test"]["mae"])<1e-4
    persistence=original[249:-1]
    assert abs(float(np.sqrt(mean_squared_error(truth,persistence)))-results["persistence_baseline_test"]["rmse"])<1e-4
    for architecture in ("LSTM","GRU"):
        scores=np.array([r["test"]["rmse"] for r in results["results"] if r["architecture"]==architecture])
        assert np.isclose(np.mean(scores),results["aggregates"][architecture]["rmse"]["mean"],rtol=1e-6)
        assert np.isclose(np.std(scores,ddof=1),results["aggregates"][architecture]["rmse"]["sample_sd"],rtol=1e-6)
    return {"dataset":309,"train":178,"validation":60,"test":59,"model_runs":6,
            "heldout_forecasts":354,"status":"NUMERICALLY_VERIFIED"}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("results_json")
    print(json.dumps(verify(parser.parse_args().results_json),indent=2))
