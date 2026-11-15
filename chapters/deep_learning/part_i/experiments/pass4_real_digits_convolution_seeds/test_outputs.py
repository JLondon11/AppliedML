"""Independently verify original Digits source, fixed-kernel maps, all 12 runs.

Run after run.py:
  python test_outputs.py --out outputs
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.ndimage import correlate
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score

def verify(folder):
    folder=Path(folder)
    j=json.loads((folder/"real_digits_seed_evidence.json").read_text())
    d=load_digits(); images=d.images.astype(np.float64); label=d.target
    assert j["observations"]==1797
    assert hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()==j["image_pixels_sha256_float64"]
    assert np.array_equal(j["image_pixels"],images[j["image_index"]])
    for name,coefficients in j["kernels"].items():
        original=correlate(images[j["image_index"]],np.asarray(coefficients),mode="reflect")
        assert np.array_equal(original,j["response_maps"][name])
        assert np.isclose(abs(original).mean(),j["mean_absolute_responses"][name])
    tr=np.asarray(j["train_indices"]);te=np.asarray(j["test_indices"])
    assert (len(tr),len(te))==(1347,450)
    assert not (set(tr)&set(te))
    assert np.array_equal(j["test_ground_truth"],label[te])
    scores=[];f1s=[]
    for record in j["run_metrics"]:
        key=str(record["seed"]); pred=np.asarray(j["run_predictions"][key])
        assert pred.shape==(450,)
        assert np.isclose(accuracy_score(label[te],pred),record["accuracy"])
        assert np.isclose(f1_score(label[te],pred,average="macro"),record["macro_f1"])
        assert len(j["run_training_losses"][key])==record["epochs_run"]
        scores.append(record["accuracy"]);f1s.append(record["macro_f1"])
    assert len(scores)==12 and len(set(j["seeds"]))==12
    assert np.isclose(np.mean(scores),j["accuracy_mean"])
    assert np.isclose(np.std(scores,ddof=1),j["accuracy_sd_sample"])
    assert np.isclose(min(scores),j["accuracy_min"])
    assert np.isclose(max(scores),j["accuracy_max"])
    assert np.isclose(np.mean(f1s),j["macro_f1_mean"])
    for stem in ("Figure_2_19_convolution_responses","Figure_2_35_seed_variation"):
        for ext in ("png","pdf","svg"):
            assert (folder/(stem+"."+ext)).stat().st_size>10000
    return {"status":"PASS","original_image_records":1797,"heldout_cases":450,
            "original_pixel_response_maps":3,"executed_models":12,"verified_predictions":5400}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="outputs")
    print(json.dumps(verify(p.parse_args().out),indent=2))
