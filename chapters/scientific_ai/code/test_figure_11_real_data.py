"""Integrity test for real 8-thermocouple Scientific AI Figure 11 outputs.

Run after the experiment script:
 python test_figure_11_real_data.py --results outputs/results.json
The test never invents measurements or predictions and does not assume that
a physics regularizer must improve measured performance.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def verify(path):
    p=Path(path); j=json.loads(p.read_text())
    measurements=np.asarray(j["recorded_temperatures_c"],dtype=np.float64)
    times=np.asarray(j["recorded_times_s"],dtype=np.float64)
    assert measurements.shape==(168,8)
    assert (len(times),times[0],times[-1])==(168,0,1670)
    assert np.allclose(np.diff(times),10)
    assert j["n_original_measurements"]==1344
    observed_digest=hashlib.sha256(measurements.astype("<f8").tobytes()).hexdigest()
    assert observed_digest=="1e8972a871c8b45ef0a2a646611e75b9b151e1bb5fb6d4998deecfa272e0154f"
    assert set(j["train_channels"]).isdisjoint({j["validation_channel"],j["test_channel"]})
    assert j["validation_channel"]!=j["test_channel"]
    assert j["training_observations"]==1008 and j["testing_observations"]==168
    assert j["validation_observations"]==168
    rows=j["results"]
    assert len(rows)==3
    for row in rows:
        output=np.asarray(j["predictions_c_by_weight"][str(row["physics_weight"])])
        assert output.shape==measurements.shape
        for sensor,key in [(2,"validation_rmse_c"),(5,"heldout_test_rmse_c")]:
            score=float(np.sqrt(np.mean((measurements[:,sensor]-output[:,sensor])**2)))
            assert np.isclose(row[key],score,rtol=1e-10)
    chosen=min(rows,key=lambda v:v["validation_rmse_c"])["physics_weight"]
    assert chosen==j["selected_weight_by_validation"]
    assert chosen==0.0  # measured result, NOT a prescribed desired winner
    return {"validated_measured_records":1344,"model_variants":3,
            "selected_weight":chosen,"test_rmse_c":rows[0]["heldout_test_rmse_c"]}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.results),indent=2))
