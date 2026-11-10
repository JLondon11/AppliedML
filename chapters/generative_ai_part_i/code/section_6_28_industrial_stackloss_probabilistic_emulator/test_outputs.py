"""Recompute saved industrial emulator evidence; run after run.py."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from statsmodels.datasets import stackloss
from sklearn.metrics import mean_squared_error,mean_absolute_error
ROOT=Path(__file__).resolve().parent/"results"
m=json.loads((ROOT/"manifest.json").read_text())
data=stackloss.load_pandas().data
x=data[["STACKLOSS","AIRFLOW","WATERTEMP","ACIDCONC"]].to_numpy(dtype="<f8")
assert hashlib.sha256(x.tobytes()).hexdigest()==m["source_array_sha256"]
out=pd.read_csv(ROOT/"heldout_predictions.csv")
draws=pd.read_csv(ROOT/"conditional_generated_samples.csv")
assert len(out)==21 and len(draws)==2100
assert list(out.heldout_id)==list(range(1,22))
assert np.allclose(out.observed_stack_loss,x[:,0])
assert list(draws.groupby("heldout_id").size())==[100]*21
assert ((draws.model_generated_stack_loss<0).sum())==2
assert int(out.interval_covers_observation.sum())==19
for name,col in [("Gaussian-process emulator","gp_predictive_mean"),("Ridge regression","ridge_prediction"),("Training-mean baseline","training_mean_prediction")]:
    assert abs(mean_absolute_error(out.observed_stack_loss,out[col])-m["metrics"][name]["MAE"])<1e-8
    assert abs(np.sqrt(mean_squared_error(out.observed_stack_loss,out[col]))-m["metrics"][name]["RMSE"])<1e-8
for f in ("stackloss_heldout_predictions.pdf","stackloss_baseline_benchmark.pdf",
          "stackloss_heldout_predictions.png","stackloss_baseline_benchmark.png"):
    assert (ROOT/f).stat().st_size>10000
print("PASS: real checksum, 21 held-out predictions, 2,100 generated samples, measured scores, physics violations and plots")
