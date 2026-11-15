"""Figure 5.50: plot original CrowdHuman CPU Faster R-CNN experimental counts.
Source execution: https://github.com/JLondon11/AppliedML/actions/runs/42322485733
This visualization contains no simulated detections or published benchmark scores.
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True,help="Artifact recall_by_occlusion.csv")
    p.add_argument("--out",type=Path,default=Path("crowdhuman_occlusion_recall.pdf"))
    args=p.parse_args()
    with args.input.open(newline="") as f:
        rows=list(csv.DictReader(f))
    expected={"low":(157,189),"medium":(81,142),"heavy":(21,145)}
    found={r["stratum"]:(int(r["matched"]),int(r["total"])) for r in rows}
    if found != expected:
        raise ValueError("Counts differ from audited original CPU experiment; rerun evidence audit")
    labels=["low","medium","heavy"]
    rates=[found[k][0]/found[k][1] for k in labels]
    fig,ax=plt.subplots(figsize=(7.2,4.35),layout="constrained")
    bars=ax.bar([k.title() for k in labels],rates,width=.58,color=["#167994","#9c7b56","#bd615a"])
    for bar,key,rate in zip(bars,labels,rates):
        hit,total=found[key]
        ax.text(bar.get_x()+bar.get_width()/2,rate+.025,f"{rate:.3f} ({hit}/{total})",
                ha="center",fontsize=10)
    ax.set(xlabel="Occlusion proxy category",
           ylabel="Full-body detection recall at IoU ≥ 0.50",ylim=(0,1.08))
    ax.grid(axis="y",alpha=.18)
    ax.set_axisbelow(True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=240,bbox_inches="tight",facecolor="white")
    plt.close(fig)

if __name__=="__main__":
    main()
