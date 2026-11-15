"""Scientific results figure for executed 170-image original PennFudanPed detector run.

Input: GitHub Actions artifact from run 42458502527.
Outputs: measured PDF/SVG/PNG. Fails closed if provenance/prediction counts
differ from the recorded original full-data experiment.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    m={row["metric"]:float(row["value"]) for row in csv.DictReader((a.evidence/"metrics.csv").open())}
    provenance=json.loads((a.evidence/"provenance.json").read_text())
    predictions=list(csv.DictReader((a.evidence/"detections.csv").open()))
    assert provenance["evaluated_images"] == 170
    assert len(provenance["sample_hashes"])==170
    assert len({x["image"] for x in provenance["sample_hashes"]})==170
    assert len(predictions)==int(m["predicted_boxes"])==1199
    assert int(m["ground_truth_persons"])==423
    assert int(m["true_positives"])==417
    assert abs(m["true_positives"]/m["ground_truth_persons"]-m["recall_at_last_prediction"])<1e-12
    assert abs(m["true_positives"]/m["predicted_boxes"]-m["precision_at_last_prediction"])<1e-12
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),layout="constrained")
    values=[m["true_positives"],m["predicted_boxes"]-m["true_positives"],m["ground_truth_persons"]-m["true_positives"]]
    labels=["Matched detections","Unmatched detections","Missed people"]
    bars=axes[0].barh(labels,values,color=["#267D87","#B27158","#948877"],height=.58)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Count from 170 original photographs")
    axes[0].set_xlim(0,950)
    for bar,v in zip(bars,values):
        axes[0].text(v+12,bar.get_y()+bar.get_height()/2,str(int(v)),va="center",fontsize=10)
    scores=[m["local_ap50"],m["precision_at_last_prediction"],m["recall_at_last_prediction"]]
    bars=axes[1].barh(["Local AP50","Precision","Recall"],scores,
                     color=["#267D87","#B27158","#4F788B"],height=.58)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0,1.14)
    axes[1].set_xlabel("Measured metric (0–1)")
    for bar,v in zip(bars,scores):
        axes[1].text(v+.02,bar.get_y()+bar.get_height()/2,f"{v:.3f}",va="center",fontsize=10)
    for axis in axes:
        axis.grid(axis="x",alpha=.12)
        axis.set_axisbelow(True)
        axis.spines[["top","right"]].set_visible(False)
    a.output.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","png","svg"):
        fig.savefig(a.output/f"pennfudan_full170_measured.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)
    print("Verified 170 original image hashes and 1,199 predictions; figure generated")

if __name__=="__main__":
    main()
