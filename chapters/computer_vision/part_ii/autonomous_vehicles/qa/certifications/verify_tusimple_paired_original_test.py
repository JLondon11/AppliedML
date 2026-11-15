"""Fail-closed audit of genuine original TuSimple paired U-Net / HSV run artifacts.

Usage: python verify_tusimple_paired_original_test.py unet-run.zip paired-run.zip
Unet run: https://github.com/JLondon11/AppliedML/actions/runs/42432540622
Paired baseline run: https://github.com/JLondon11/AppliedML/actions/runs/42526905776
"""
import csv,hashlib,io,json,sys,zipfile
import numpy as np

if len(sys.argv)!=3: raise SystemExit("Provide both archived original GitHub results")
with zipfile.ZipFile(sys.argv[1]) as old,zipfile.ZipFile(sys.argv[2]) as new:
    original=list(csv.DictReader(io.StringIO(old.read("per_frame.csv").decode())))
    rows=list(csv.DictReader(io.StringIO(new.read("per_image_comparison.csv").decode())))
    original_summary=json.loads(old.read("metrics.json"))
    result=json.loads(new.read("metrics.json"))
assert len(original)==len(rows)==32
assert len({r["frame"] for r in rows})==32
assert len({r["image_sha256"] for r in rows})==32
assert result["evaluation"]["paired_original_photo_sha256_verified"] is True
assert result["evaluation"]["checkpoint_training_membership_verified"] is False
for a,b in zip(rows,original):
    assert a["frame"]==b["frame"]
    assert a["image_sha256"]==b["image_sha256"]
    assert int(a["gt_pixels"])==int(b["gt_pixels"])
    for field in ("tp","fp","fn"):
        assert int(a["unet_"+field])==int(b[field])
    for model in ("unet","baseline"):
        tp,fp,fn=[int(a[model+"_"+v]) for v in ("tp","fp","fn")]
        assert tp+fn==int(a["gt_pixels"])
        assert abs(tp/max(1,tp+fp+fn)-float(a[model+"_iou"]))<1e-14
        assert abs(2*tp/max(1,2*tp+fp+fn)-float(a[model+"_dice"]))<1e-14
for model in ("unet","baseline"):
    tp,fp,fn=[sum(int(r[model+"_"+k]) for r in rows) for k in ("tp","fp","fn")]
    assert abs(result[model]["micro_iou"]-tp/(tp+fp+fn))<1e-14
    assert abs(result[model]["micro_dice"]-2*tp/(2*tp+fp+fn))<1e-14
assert abs(result["unet"]["micro_iou"]-original_summary["micro_iou"])<1e-14
assert abs(result["unet"]["micro_dice"]-original_summary["micro_dice"])<1e-14
assert sum(float(r["unet_iou"])>float(r["baseline_iou"]) for r in rows)==32
rng=np.random.default_rng(20261108)
delta=[]
for _ in range(5000):
    picked=[rows[int(i)] for i in rng.integers(0,32,size=32)]
    aggregated=[]
    for model in ("unet","baseline"):
        tp,fp,fn=[sum(int(r[model+"_"+k]) for r in picked) for k in ("tp","fp","fn")]
        aggregated.append(tp/(tp+fp+fn))
    delta.append(aggregated[0]-aggregated[1])
interval=list(map(float,np.quantile(delta,[.025,.975])))
assert np.allclose(interval,result["delta_micro_iou_bootstrap_ci95"],atol=1e-14,rtol=0)
print(json.dumps({"status":"PASS","original_photographs":32,
    "equal_source_checksums":32,"unet_wins":32,
    "micro_iou_difference":result["delta_unet_minus_baseline_micro_iou"],
    "image_paired_bootstrap_95ci":interval,
    "original_unet_artifact_sha256":hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest(),
    "paired_artifact_sha256":hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest()},indent=2))
