"""Independent source-to-prediction check for the 144 original AirPassengers counts."""
import csv,hashlib,json,argparse
from pathlib import Path
import numpy as np
def verify(data,folder):
 out=Path(folder);data=Path(data)
 j=json.loads((out/"measured_airpassengers_forecast.json").read_text())
 with data.open() as f:rows=list(csv.DictReader(f))
 original=np.array([int(r["Passengers"]) for r in rows])
 assert len(original)==144 and original.sum()==40363
 assert j["local_file_sha256"]==hashlib.sha256(data.read_bytes()).hexdigest()
 assert j["original_observations"]==original.tolist()
 assert (j["train_windows"],j["validation_windows"],j["test_windows"])==(84,24,24)
 assert j["test_target_months"]==[r["Month"] for r in rows][120:144]
 actual=original[120:144]
 assert np.array_equal(actual,j["test_targets"])
 predictions=j["predictions"]
 expected={"seasonal_one_step":original[108:132],
  "persistence_one_step":original[119:143],
  "seasonal_recursive_24_month":np.tile(original[108:120],2),
  "persistence_recursive_24_month":np.full(24,original[119])}
 for name,values in expected.items():assert np.array_equal(values,predictions[name])
 for key,p in predictions.items():
  p=np.asarray(p)
  assert p.shape==(24,) and np.isfinite(p).all()
  m=j["scores"][key]
  assert np.isclose(np.sqrt(np.mean((actual-p)**2)),m["rmse"],atol=1e-4)
  assert np.isclose(np.mean(np.abs(actual-p)),m["mae"],atol=1e-4)
  assert np.isclose(np.mean(p-actual),m["bias"],atol=1e-4)
 for e in ("png","pdf","svg"):
  assert (out/("Figure_2_AirPassengers_RealData_LSTM_Forecast."+e)).stat().st_size>10000
 return {"PASS":True,"source_observations":144,"test_months":24,
         "actual_prediction_methods":len(predictions),"recreated_scientific_figures":3}
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--data",required=True);p.add_argument("--output",required=True)
 a=p.parse_args();print(json.dumps(verify(a.data,a.output),indent=2))
