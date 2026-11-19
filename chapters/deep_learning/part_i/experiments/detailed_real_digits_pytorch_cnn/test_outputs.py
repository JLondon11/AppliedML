"""Verify original-data CNN evidence and genuine per-image prediction metrics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score,f1_score,log_loss,confusion_matrix

def verify(out):
    out=Path(out)
    r=json.loads((out/"original_digits_pytorch_cnn_test_evidence.json").read_text())
    d=load_digits()
    X=d.images.astype(np.float32)/16.
    y=d.target.astype(np.int64)
    assert hashlib.sha256(X.tobytes()).hexdigest()==r["source_float32_sha256"]
    tr,va,te=[np.asarray(r[k]) for k in ("train_indices","validation_indices","test_indices")]
    assert (len(tr),len(va),len(te))==(1077,270,450)
    assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    actual=y[te]
    pred=np.asarray(r["test_CNN_predict"])
    base=np.asarray(r["test_logistic_predict"])
    prob=np.asarray(r["test_CNN_probabilities"])
    assert prob.shape==(450,10) and np.allclose(prob.sum(axis=1),1,atol=1e-5)
    assert np.array_equal(actual,r["test_labels"])
    assert np.isclose(accuracy_score(actual,pred),r["cnn_accuracy"])
    assert np.isclose(f1_score(actual,pred,average="macro"),r["cnn_macro_f1"])
    assert np.isclose(log_loss(actual,prob/prob.sum(axis=1,keepdims=True),labels=np.arange(10)),r["cnn_test_cross_entropy"],rtol=1e-3)
    assert np.isclose(accuracy_score(actual,base),r["logistic_baseline_test_accuracy"])
    assert np.array_equal(confusion_matrix(actual,pred),r["confusion_matrix"])
    assert np.sum(actual!=pred)==r["number_of_test_errors"]==6
    for ext in ("png","pdf","svg"):
        assert (out/("Figure_2_Observed_Digits_PyTorch_CNN."+ext)).stat().st_size>10000
    return {"PASS":True,"original_images":1797,"heldout_evaluations":450}
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="outputs")
    print(verify(parser.parse_args().out))
