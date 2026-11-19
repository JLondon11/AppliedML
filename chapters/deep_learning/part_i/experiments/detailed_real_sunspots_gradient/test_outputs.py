"""Verify the 18 executed initial-gradient runs against original real sunspot data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from statsmodels.datasets import sunspots

def verify(out):
    out=Path(out)
    evidence=json.loads((out/"original_sunspot_gradient_evidence.json").read_text())
    original=sunspots.load_pandas().data.SUNACTIVITY.to_numpy(dtype=np.float32)
    assert len(original)==309
    assert hashlib.sha256(original.tobytes()).hexdigest()==evidence["source_sha256_float32"]
    assert len(evidence["records"])==18
    for n in (12,36):
        for name in ("RNN","LSTM","GRU"):
            records=[r for r in evidence["records"] if r["window"]==n and r["model"]==name]
            assert len(records)==3 and {r["seed"] for r in records}=={17,42,84}
            for row in records:
                observed=np.asarray(row["input_abs_gradient_by_training_window"])
                assert observed.shape==(24,n)
                median=np.median(observed,axis=0)
                assert np.allclose(median,row["median_absolute_gradient_lag_1_to_N"])
                assert np.isclose(median[-1]/max(median[0],1e-20),row["earliest_to_latest_ratio"])
    for ext in ("png","pdf","svg"):
        assert (out/("Figure_2_Observed_Sunspots_Recurrent_Input_Gradient."+ext)).stat().st_size>10000
    return {"PASS":True,"original_observations":309,"initial_gradient_model_runs":18}
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="outputs")
    print(verify(parser.parse_args().out))
