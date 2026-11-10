"""Verify all Figure 11.22 measured outputs; no unsupported accuracy claims."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT=Path(__file__).resolve().parent

def verify():
    subprocess.run([sys.executable,str(ROOT/"run.py")],cwd=ROOT,check=True)
    j=json.loads((ROOT/"outputs/measured_results.json").read_text())
    assert j["image_shape"]==[512,512,3]
    assert j["candidate_regions_count"]==len(j["candidate_region_areas_px"])
    assert j["candidate_regions_count"]==len(j["candidate_region_eccentricities"])
    assert j["segmentation_has_ground_truth"] is False
    assert j["segmentation_dice"] is None and j["segmentation_aji"] is None
    assert j["classification_has_pairing_to_photograph"] is False
    assert (j["classification_train"],j["classification_val"],
            j["classification_test"])==(341,114,114)
    y=np.asarray(j["classification_test_ground_truth"])
    probs=np.asarray(j["classification_test_probabilities"])
    assert len(y)==len(probs)==114
    assert np.isclose(roc_auc_score(y,probs),j["classification_test_auc"],rtol=1e-12)
    assert np.isclose(accuracy_score(y,probs>=.5),
                      j["classification_test_accuracy"],rtol=1e-12)
    stem="Figure_11_22_Watershed_Nuclear_Segmentation_and_Classification"
    for ext in ("png","pdf","svg"):
        assert (ROOT/"outputs"/f"{stem}.{ext}").stat().st_size>1000

if __name__=="__main__":
    verify()
    print("PASS: actual microscopy candidates and independent WDBC predictions. MoNuSeg NOT certified.")
