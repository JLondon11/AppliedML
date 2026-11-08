"""Fail-closed independent audit of archived paired CrowdHuman original-data results.

Usage: python verify_crowdhuman_paired.py ARTIFACT_DIRECTORY
Requires per_person_paired.csv and metrics.json from successful Actions model run.
Independently recomputes all counts, architecture comparisons, and 5000 paired
image-cluster bootstrap percentile intervals.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def main(root):
    root=Path(root)
    with (root/"per_person_paired.csv").open(newline="") as f:rows=list(csv.DictReader(f))
    with (root/"metrics.json").open() as f:report=json.load(f)
    assert len(rows)==3178 and report["images"]==48 and report["reference_people"]==1589
    models=("MobileNetV3-FPN","ResNet50-FPN")
    bins=("overall","low","medium","heavy")
    index={model:{(r["image"],int(r["instance"])):r for r in rows if r["model"]==model} for model in models}
    assert len(index[models[0]])==len(index[models[1]])==1589
    assert set(index[models[0]])==set(index[models[1]])
    keys=sorted({k[0] for k in index[models[0]]})
    assert len(keys)==48
    digest_by_image={}
    for key in index[models[0]]:
        left=index[models[0]][key]
        right=index[models[1]][key]
        assert left["occlusion_bin"]==right["occlusion_bin"]
        assert left["image_sha256"]==right["image_sha256"]
        oc=float(left["occlusion_area_proxy"])
        assert 0<=oc<=1
        assert left["occlusion_bin"]==("low" if round(oc,12)<.2 else "medium" if round(oc,12)<.5 else "heavy")
        assert all(int(x["matched_IoU50"]) in (0,1) for x in (left,right))
        if key[0] in digest_by_image:assert digest_by_image[key[0]]==left["image_sha256"]
        digest_by_image[key[0]]=left["image_sha256"]
    assert len(set(digest_by_image.values()))==48
    counts={}
    for img in keys:
        for b in bins:
            instances=[key for key,row in index[models[0]].items()
                if key[0]==img and (b=="overall" or row["occlusion_bin"]==b)]
            n=len(instances)
            a=sum(int(index[models[0]][i]["matched_IoU50"]) for i in instances)
            c=sum(int(index[models[1]][i]["matched_IoU50"]) for i in instances)
            counts[(img,b)]=(n,a,c)
    for b in bins:
        n=sum(counts[(im,b)][0] for im in keys)
        for name,k in zip(models,(1,2)):
            hits=sum(counts[(im,b)][k] for im in keys)
            r=report["metrics"][name][b]
            assert (hits,n)==(r["matched"],r["count"])
            assert abs(hits/n-r["recall"])<1e-12
    rng=np.random.default_rng(20261108)
    draws=rng.integers(0,48,size=(5000,48))
    ci={}
    for b in bins:
        v=np.array([counts[(im,b)] for im in keys])
        sample=v[draws]
        diff=(sample[:,:,2].sum(1)-sample[:,:,1].sum(1))/sample[:,:,0].sum(1)
        ci[b]=list(map(float,np.quantile(diff,[.025,.975])))
        assert np.allclose(ci[b],report["paired_difference_resnet_minus_mobilenet_CI95"][b],
                           atol=1e-12,rtol=0)
    return {"result":"PASS","original_images":48,"paired_annotated_people":1589,
       "detector_result_rows":len(rows),"paired_bootstrap_intervals":ci,
       "per_person_csv_sha256":hashlib.sha256((root/"per_person_paired.csv").read_bytes()).hexdigest()}

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("artifact_directory")
    print(json.dumps(main(a.parse_args().artifact_directory),indent=2))
