#!/usr/bin/env python3
"""Chapter 2 Pass 6 evidence verification for the *full distributed release ZIP*.

This is a source-to-saved-output gate, not retraining. Use:
  python verify_ch2_distributed_science.py /path/to/Chapter2_Pass6...zip
Requires numpy, scipy, scikit-learn; checks archived original-data predictions
and authentic image response maps. It does not assume unpublished GitHub assets.
"""
import argparse, hashlib, json, tempfile, zipfile
from pathlib import Path
import numpy as np
from scipy.ndimage import correlate
from sklearn.datasets import load_digits,load_diabetes
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

def verify(archive):
    archive=Path(archive)
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        root="Chapter2_Pass3/"
        def load(name):
            return json.loads(z.read(root+name))
        seeds=load("code/real_digits_seed_evidence.json")
        binary=load("code/real_digits_nonlinear_2d_evidence.json")
        both=load("evidence/measured_results.json")
        digits=load_digits()
        X=digits.data.astype(np.float64)/16
        assert len(X)==1797
        assert hashlib.sha256(X.tobytes()).hexdigest()==binary["dataset_sha256_float64"]
        selected=np.asarray(binary["test_source_indices"])
        y=(digits.target[selected]==8).astype(int)
        assert len(y)==108 and np.array_equal(y,binary["test_ground_truth"])
        assert np.isclose(accuracy_score(y,binary["mlp_test_predict"]),binary["mlp_accuracy"])
        assert np.isclose(accuracy_score(y,binary["baseline_test_predict"]),binary["baseline_accuracy"])
        assert np.isclose(roc_auc_score(y,binary["mlp_test_probabilities"]),binary["mlp_roc_auc"])
        pixels=digits.images.astype(np.float64)
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==seeds["image_pixels_sha256_float64"]
        for name,kernel in seeds["kernels"].items():
            a=correlate(pixels[seeds["image_index"]],np.asarray(kernel),mode="reflect")
            assert np.array_equal(a,seeds["response_maps"][name])
        test=np.asarray(seeds["test_indices"])
        scores=[]
        for run in seeds["run_metrics"]:
            p=seeds["run_predictions"][str(run["seed"])]
            assert len(p)==450
            assert np.isclose(accuracy_score(digits.target[test],p),run["accuracy"])
            assert np.isclose(f1_score(digits.target[test],p,average="macro"),run["macro_f1"])
            scores.append(run["accuracy"])
        assert len(scores)==12
        assert np.isclose(np.mean(scores),seeds["accuracy_mean"])
        assert np.isclose(np.std(scores,ddof=1),seeds["accuracy_sd_sample"])
        d=both["digit_study"];b=both["diabetes_study"]
        yt=digits.target[d["test_source_indices"]]
        assert hashlib.sha256(X.tobytes()).hexdigest()==d["source_sha256_float64_normalized"]
        assert len(yt)==450 and np.array_equal(yt,d["test_true"])
        assert np.isclose(accuracy_score(yt,d["test_pred"]),d["accuracy"])
        assert np.isclose(roc_auc_score(yt,d["test_probabilities"],multi_class="ovr"),d["macro_ovr_auc"])
        z=load_diabetes()
        assert hashlib.sha256(z.data.astype(np.float64).tobytes()).hexdigest()==b["source_sha256_float64"]
        actual=z.target[b["test_source_indices"]]
        assert len(actual)==111 and np.array_equal(actual,b["test_actual"])
        assert np.isclose(np.sqrt(mean_squared_error(actual,b["test_pred"])),b["rmse"])
        assert np.isclose(np.sqrt(mean_squared_error(actual,b["test_baseline_pred"])),b["baseline_rmse"])
        for stem in ("Figure_2_05_two_moons_boundary","Figure_2_19_convolution_responses",
                     "Figure_2_31_digits_training_comparison","Figure_2_32_digits_error_structure",
                     "Figure_2_33_digits_calibration_errors","Figure_2_34_diabetes_regression_diagnostics",
                     "Figure_2_35_seed_variation"):
            for ext in ("png","pdf","svg"):
                assert root+f"figures/{stem}.{ext}" in names
        assert root+"qa_code/verify_measured_case_studies.py" in names
        assert root+"qa_code/verify_ch2_scientific_evidence.py" in names
        return {"status":"PASS","scientific_scope_levels":{"A":2,"B":3},
                "verified_output_counts":{"binary_classification":108,
                "digits_classification":450,"diabetes_regression":111,
                "multi_seed_classification":5400},"empirical_asset_files":21,
                "limitation":"saved original-data output integrity; not model retraining or production certification"}

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("release_zip")
    print(json.dumps(verify(a.parse_args().release_zip),indent=2))
