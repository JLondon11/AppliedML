"""Scientific QA for the original-photograph, single-step denoising ablation.

Uses the saved genuine-image experiment output from run.py --steps 65.
Run: python verify_scoped_restoration.py /path/to/results
This is NOT an iterative DDPM or scientific microscopy certificate.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from skimage import data, color, transform

SEEDS=(20261106,20261107,20261108)
TRAIN=("astronaut","coffee","chelsea","rocket")
TEST=("camera","coins")
ALPHAS=(0.90,0.75)
ARMS=("noisy","gaussian_sigma_0.8","unconditioned","conditioned")
EXPECTED={
    "noisy":(9.928145475294286,0.1291182247448632),
    "gaussian_sigma_0.8":(16.988781442011234,0.3177180366962218),
    "unconditioned":(17.352023524825412,0.2906048160443729),
    "conditioned":(17.68407455309745,0.3081990580058902),
}

def original_gray(im):
    if im.ndim==3:
        im=color.rgb2gray(im[:,:,:3])
    if np.issubdtype(im.dtype,np.integer):
        im=im.astype(np.float32)/np.iinfo(im.dtype).max
    return transform.resize(np.asarray(im,dtype=np.float32),(128,128),
                            anti_aliasing=True).astype(np.float32)

def audit(out):
    out=Path(out)
    m=json.loads((out/"manifest.json").read_text())
    if m.get("steps_per_model")!=65 or tuple(m.get("seeds",()))!=SEEDS:
        raise AssertionError("Expected three full 65-step real training seeds")
    if tuple(m["training_images"])!=TRAIN or tuple(m["heldout_test_images"])!=TEST:
        raise AssertionError("Original train/held-out photograph inventory mismatch")
    if set(TRAIN)&set(TEST):
        raise AssertionError("Original photograph leakage")
    fingerprints=m.get("image_sha256",m.get("sha256",{}))
    for name in TRAIN+TEST:
        raw=original_gray(getattr(data,name)())
        digest=hashlib.sha256(raw.tobytes()).hexdigest()
        if digest!=fingerprints.get(name):
            raise AssertionError("Original real photo fingerprint mismatch: "+name)
    with (out/"metrics.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=48:
        raise AssertionError("48 repeated experimental condition evaluations expected")
    counts=Counter((int(r["seed"]),r["image"],float(r["alpha_bar"]),r["model"]) for r in rows)
    expected={(s,im,alpha,arm) for s in SEEDS for im in TEST for alpha in ALPHAS for arm in ARMS}
    if set(counts)!=expected or any(value!=1 for value in counts.values()):
        raise AssertionError("Missing, repeated or invented experimental condition")
    arm_scores=defaultdict(list)
    for r in rows:
        psnr=float(r["psnr_db"]);ssim=float(r["ssim"])
        if not math.isfinite(psnr) or not math.isfinite(ssim) or not (-1<=ssim<=1):
            raise AssertionError("Invalid original-image PSNR / SSIM")
        arm_scores[r["model"]].append((psnr,ssim))
    means={}
    for arm in ARMS:
        psnr=float(np.mean([p for p,s in arm_scores[arm]]))
        ssim=float(np.mean([s for p,s in arm_scores[arm]]))
        if abs(psnr-EXPECTED[arm][0])>0.0001 or abs(ssim-EXPECTED[arm][1])>0.0001:
            raise AssertionError(f"Replayed original-image benchmark mismatch: {arm}")
        means[arm]={"mean_psnr_db":psnr,"mean_ssim":ssim}
    with (out/"training.csv").open(newline="") as f:
        trace=list(csv.DictReader(f))
    if len(trace)!=390:
        raise AssertionError("6 original training histories x 65 steps expected")
    trace_counts=Counter((int(x["seed"]),x["model"]) for x in trace)
    if trace_counts!={(s,arm):65 for s in SEEDS for arm in ("conditioned","unconditioned")}:
        raise AssertionError("Incomplete original training traces")
    for seed in SEEDS:
        for arm in ("conditioned","unconditioned"):
            file=out/f"model_{arm}_seed{seed}.pt"
            if file.stat().st_size<10000:
                raise AssertionError("Actual trained checkpoint missing: "+file.name)
    for name in ("real_photo_ablation_outputs.png","ablation_psnr.png","ablation_training.png"):
        if (out/"figures"/name).stat().st_size<10000:
            raise AssertionError("Data-derived figure missing: "+name)
    # One should not count correlated repeats as 48 original photographs.
    return {"status":"SCOPED_EXPERIMENT_QA_PASS","original_train_photos":4,
            "independent_heldout_original_photos":2,"training_runs":6,
            "condition_evaluations":48,"distinct_heldout_images":2,
            "means":means,"certified_scope":"single-step photographic noise-prediction ablation only"}

if __name__=="__main__":
    import sys
    print(json.dumps(audit(sys.argv[1] if len(sys.argv)>1 else
                           Path(__file__).parent/"results"),indent=2))
