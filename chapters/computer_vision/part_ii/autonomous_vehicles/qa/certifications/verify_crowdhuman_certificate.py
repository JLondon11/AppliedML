"""Independently verify measured CrowdHuman occlusion certificate from run artifact ZIP.

Example:
  python verify_crowdhuman_certificate.py chapter5_verified_13554287376.zip
This is not an inference runner; it checks evidence from an already executed model.
"""
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict

import numpy as np

artifact=sys.argv[1]
with zipfile.ZipFile(artifact) as archive:
    names=set(archive.namelist())
    assert {"run_manifest.json","per_person_results.csv","recall_by_occlusion.csv",
            "crowdhuman_occlusion_recall.pdf"} <= names
    manifest=json.loads(archive.read("run_manifest.json"))
    rows=list(csv.DictReader(io.StringIO(archive.read("per_person_results.csv").decode())))
    summary=list(csv.DictReader(io.StringIO(archive.read("recall_by_occlusion.csv").decode())))

assert manifest["processed_images"]==len({r["image"] for r in rows})==96
assert manifest["person_instances"]==len(rows)==2296
assert len(manifest["checkpoint_sha256"])==64
targets={"low":(653,861),"medium":(325,638),"heavy":(92,797)}
for b,(hits,count) in targets.items():
    subset=[r for r in rows if r["occlusion_bin"]==b]
    assert len(subset)==count
    assert sum(int(r["matched_IoU50"]) for r in subset)==hits
    for r in subset:
        oc=float(r["occlusion_area_proxy"])
        assert 0 <= oc <= 1.00001
        assert (oc < .2 if b=="low" else .2<=oc<.5 if b=="medium" else oc>=.5)
        assert float(r["x2"])>float(r["x1"])
        assert float(r["y2"])>float(r["y1"])
        assert int(r["matched_IoU50"]) in (0,1)
for r in summary:
    h,n=targets[r["stratum"]]
    assert (int(r["matched"]),int(r["total"]))==(h,n)
    assert abs(float(r["local_recall_IoU50"])-h/n)<1e-12

groups=defaultdict(list)
for r in rows: groups[r["image"]].append(r)
keys=sorted(groups)
rng=np.random.default_rng(20261107)
draws=defaultdict(list)
def recall(sample,cat=None):
    match=[int(r["matched_IoU50"]) for r in sample if cat is None or r["occlusion_bin"]==cat]
    assert match
    return sum(match)/len(match)
for _ in range(5000):
    sample=[r for ix in rng.integers(0,len(keys),size=len(keys)) for r in groups[keys[ix]]]
    metrics={b:recall(sample,b) for b in targets}
    metrics["overall"]=recall(sample)
    metrics["low_minus_heavy"]=metrics["low"]-metrics["heavy"]
    for key,val in metrics.items():draws[key].append(val)
ci={k:list(map(float,np.quantile(v,[.025,.975]))) for k,v in draws.items()}
assert abs(recall(rows)-1070/2296)<1e-12
expected={"overall":(.3920824,.5606826),"low":(.6344371,.8669440),
          "medium":(.4145725,.6294841),"heavy":(.0894300,.1519995),
          "low_minus_heavy":(.5226127,.7506544)}
for k,v in expected.items():assert np.allclose(ci[k],v,rtol=0,atol=2e-6)
print(json.dumps({"status":"PASS","experiment":"CV5-EXP-CROWDHUMAN-96-OCCLUSION-003",
                  "photos":96,"original_person_annotations":2296,
                  "artifact_sha256":hashlib.sha256(open(artifact,"rb").read()).hexdigest(),
                  "image_cluster_bootstrap_ci95":ci},indent=2))
