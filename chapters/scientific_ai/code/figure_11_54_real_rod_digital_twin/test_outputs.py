"""Verify Chapter 11 digital-twin evidence without inventing observations.

Usage: python test_outputs.py --evidence outputs/figure_11_53_measured_evidence.json
"""
import argparse,json
from pathlib import Path
import numpy as np

def verify(path):
    path=Path(path)
    data=json.loads(path.read_text())
    assert data["recorded_rows"]==168
    assert data["training_measurement_count"]==800
    assert data["validation_time_steps"]==30
    assert data["heldout_test_time_steps"]==38
    assert data["heldout_target_evaluation_values"]==152
    assert data["training_times_indices"]==[0,99]
    assert data["validation_times_indices"]==[100,129]
    assert data["test_times_indices"]==[130,167]
    assert not (set(data["observed_test_channels"]) & set(data["withheld_target_channels"]))
    assert len(data["validation_grid"])==15
    actual=np.asarray(data["test_withheld_measured_C"])
    assert actual.shape==(38,4)
    for source,key in [("test_withheld_assimilated_C","assimilated_twin"),
                       ("test_withheld_static_ridge_C","static_sensor_ridge"),
                       ("test_withheld_prior_C","one_step_dynamics_prior")]:
        estimates=np.asarray(data[source])
        assert estimates.shape==actual.shape and np.isfinite(estimates).all()
        expected=data["heldout_results"][key]
        assert np.isclose(np.sqrt(np.mean((actual-estimates)**2)),
                          expected["RMSE_C"],rtol=1e-12)
        assert np.isclose(np.mean(np.abs(actual-estimates)),
                          expected["MAE_C"],rtol=1e-12)
    stem="Figure_11_53_Bayesian_Multiscale_Digital_Twin_Assimilation"
    for ext in ("png","pdf","svg"):
        assert (path.parent/f"{stem}.{ext}").stat().st_size>10000
    return data["heldout_results"]

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",required=True)
    print(json.dumps(verify(p.parse_args().evidence),indent=2))
    print("PASS: 152 genuine test targets, measured outputs, metric arithmetic and publication assets")
