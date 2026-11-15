"""Evidence gate: ensure every published Chapter 2 number comes from saved original predictions."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits,load_diabetes
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,mean_squared_error,r2_score
def verify(output):
    output=Path(output);j=json.loads((output/"measured_results.json").read_text())
    a=j["digits"];b=j["diabetes"]
    X=load_digits().data.astype("float64")/16
    assert hashlib.sha256(X.tobytes()).hexdigest()==a["feature_sha256_float64"]
    assert a["records"]==1797 and a["train"]==1347 and a["test"]==450
    assert not set(a["train_indices"])&set(a["test_indices"])
    actual=load_digits().target[a["test_indices"]]
    pred=np.asarray(a["predicted_test"]);prob=np.asarray(a["probabilities_test"])
    assert len(actual)==450 and prob.shape==(450,10)
    assert np.array_equal(actual,a["true_test"])
    assert np.isclose(accuracy_score(actual,pred),a["accuracy"],rtol=1e-12)
    assert np.isclose(f1_score(actual,pred,average="macro"),a["macro_f1"],rtol=1e-12)
    assert np.isclose(roc_auc_score(actual,prob,multi_class="ovr"),a["ovr_auc"],rtol=1e-12)
    assert np.isclose(accuracy_score(actual,a["baseline_predicted_test"]),a["baseline_accuracy"])
    X=load_diabetes().data.astype("float64")
    assert hashlib.sha256(X.tobytes()).hexdigest()==b["feature_sha256_float64"]
    assert b["records"]==442 and b["train"]==331 and b["test"]==111
    assert not set(b["train_indices"])&set(b["test_indices"])
    actual=load_diabetes().target[b["test_indices"]]
    p=np.array(b["predicted_test"]);q=np.array(b["ridge_predicted_test"])
    assert np.array_equal(actual,b["true_test"])
    assert np.isclose(np.sqrt(mean_squared_error(actual,p)),b["rmse"],rtol=1e-12)
    assert np.isclose(np.sqrt(mean_squared_error(actual,q)),b["ridge_rmse"],rtol=1e-12)
    assert np.isclose(r2_score(actual,p),b["r2"],rtol=1e-12)
    assert np.isclose(r2_score(actual,q),b["ridge_r2"],rtol=1e-12)
    for stem in ("Figure_2_31_digits_training_comparison","Figure_2_32_digits_error_structure",
                 "Figure_2_33_digits_calibration_errors","Figure_2_34_diabetes_regression_diagnostics"):
        for ext in ("png","pdf","svg"):
            assert (output/f"{stem}.{ext}").stat().st_size>2000
    return {"status":"PASS","real_digits_test_samples":450,
            "real_diabetes_test_samples":111,"original_digit_images":1797,
            "original_diabetes_cases":442,"measured_result_figures":4}
if __name__=="__main__":
    import sys
    print(json.dumps(verify(sys.argv[1] if len(sys.argv)>1 else "outputs"),indent=2))
