"""Verify original 3-vs-8 Digits experiment and saved held-out forecasts."""
import hashlib,json,argparse
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
def verify(folder):
    folder=Path(folder)
    j=json.loads((folder/"real_digits_nonlinear_2d_evidence.json").read_text())
    data=load_digits();source=data.data.astype("float64")/16.
    assert hashlib.sha256(source.tobytes()).hexdigest()==j["dataset_sha256_float64"]
    assert j["original_records"]==1797 and j["subset_records"]==357
    train=np.asarray(j["train_source_indices"]);test=np.asarray(j["test_source_indices"])
    assert (len(train),len(test))==(249,108)
    assert not(set(train)&set(test))
    assert set(np.unique(data.target[np.r_[train,test]]))=={3,8}
    truth=(data.target[test]==8).astype(int)
    assert np.array_equal(truth,j["test_ground_truth"])
    pred=np.array(j["mlp_test_predict"])
    base=np.array(j["baseline_test_predict"])
    probability=np.array(j["mlp_test_probabilities"])
    assert len(probability)==108
    assert np.isclose(accuracy_score(truth,pred),j["mlp_accuracy"])
    assert np.isclose(accuracy_score(truth,base),j["baseline_accuracy"])
    assert np.isclose(f1_score(truth,pred),j["mlp_f1"])
    assert np.isclose(roc_auc_score(truth,probability),j["mlp_roc_auc"])
    assert np.shape(j["test_embedded_features"])==(108,2)
    for ext in ("png","pdf","svg"):
        assert (folder/("Figure_2_05_two_moons_boundary."+ext)).stat().st_size>5000
    return {"status":"PASS","original_images":357,"heldout_images":108,
            "MLP_accuracy":j["mlp_accuracy"],"logistic_accuracy":j["baseline_accuracy"]}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="outputs")
    print(verify(p.parse_args().out))
