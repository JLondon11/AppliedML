"""Image-clustered uncertainty from original, executed COCO and CrowdHuman records.

Input CSVs must come from completed CPU runs:
COCO run 42300131086; CrowdHuman run 42323774125.
No fabricated predictions. This is subset-specific uncertainty, not an official AP.
"""
import argparse,csv,json
from collections import defaultdict
import numpy as np

def read(path):
    with open(path,newline="") as f:
        return list(csv.DictReader(f))

def bootstrap(groups,stat,n=5000,seed=20261107):
    keys=sorted(groups)
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(n):
        chosen=rng.integers(0,len(keys),size=len(keys))
        rows=[row for i in chosen for row in groups[keys[i]]]
        draws.append(stat(rows))
    return list(map(float,np.quantile(draws,[.025,.975])))

def mean_sam(rows):
    return float(np.mean([float(r["sam_iou"]) for r in rows]))

def paired_difference(rows):
    valid=[r for r in rows if r["grabcut_iou"]!=""]
    if not valid: return np.nan
    return float(np.mean([float(r["sam_iou"])-float(r["grabcut_iou"]) for r in valid]))

def recall(rows,group=None):
    subset=[r for r in rows if group is None or r["occlusion_bin"]==group]
    if not subset: return np.nan
    return sum(int(r["matched_IoU50"]) for r in subset)/len(subset)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sam",required=True)
    p.add_argument("--crowd",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    sam=read(a.sam);crowd=read(a.crowd)
    gs=defaultdict(list);gc=defaultdict(list)
    for r in sam: gs[r["image_id"]].append(r)
    for r in crowd: gc[r["image"]].append(r)
    assert (len(sam),len(gs),len(crowd),len(gc))==(120,37,2296,96)
    assert abs(mean_sam(sam)-0.7822379221932622)<1e-10
    expected={"low":(653,861),"medium":(325,638),"heavy":(92,797)}
    for label,(matched,total) in expected.items():
        x=[r for r in crowd if r["occlusion_bin"]==label]
        assert (sum(int(r["matched_IoU50"]) for r in x),len(x))==(matched,total)
    result={
      "method":"5000 image-clustered bootstrap replicates; 95% percentile intervals",
      "sam_mean_iou":mean_sam(sam),
      "sam_mean_iou_ci":bootstrap(gs,mean_sam),
      "sam_grabcut_paired_difference":paired_difference(sam),
      "sam_grabcut_paired_difference_ci":bootstrap(gs,paired_difference),
      "crowd_overall_recall":recall(crowd),
      "crowd_overall_ci":bootstrap(gc,recall),
      "crowd_by_occlusion":{
        label:{"recall":recall(crowd,label),
               "ci":bootstrap(gc,lambda rows,label=label:recall(rows,label))}
        for label in ("low","medium","heavy")},
      "low_minus_heavy":recall(crowd,"low")-recall(crowd,"heavy"),
      "low_minus_heavy_ci":bootstrap(gc,lambda rows:recall(rows,"low")-recall(rows,"heavy")),
      "qualification":"Intervals conditional on selected frames and local protocols; not population/official benchmark uncertainty."
    }
    with open(a.output,"w") as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
