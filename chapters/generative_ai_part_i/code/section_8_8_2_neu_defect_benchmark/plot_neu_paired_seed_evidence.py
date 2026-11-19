"""Source-linked scientific figure for the original NEU measured paired-seed effects.

Input is 12 actual source-derived classifier-arm/seed measurements. The original
360 test photographs are reused across seeds; no additional photographs or
GAN training are invented here.

Run:
python plot_neu_paired_seed_evidence.py --csv replicated_seed_metrics_pass13.csv --out neu_paired_seed_scientific_evidence.png
"""
from __future__ import annotations
import argparse
import csv
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render(csv_path: Path, out: Path):
    with csv_path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    seeds=(20261106,20261107,20261108)
    arms=("real_only","flip_augmentation","gan_augmented","real_repeat_augmented")
    if len(rows)!=12 or {(int(r['seed']),r['arm']) for r in rows}!={(s,a) for s in seeds for a in arms}:
        raise ValueError("Missing or repeated original three-seed NEU training-arm metric")
    source={(int(x["seed"]),x["arm"]):x for x in rows}
    pairs=[("test_accuracy","real_only","Accuracy · GAN − original photos"),
          ("test_macro_f1","real_only","Macro-F1 · GAN − original photos"),
          ("test_accuracy","real_repeat_augmented","Accuracy · GAN − matched repeats"),
          ("test_macro_f1","real_repeat_augmented","Macro-F1 · GAN − matched repeats")]
    fig,ax=plt.subplots(figsize=(10,5.7))
    for i,(metric,comparator,label) in enumerate(pairs):
        d=[100*(float(source[s,"gan_augmented"][metric])-float(source[s,comparator][metric]))
           for s in seeds]
        if not all(-43 < value < 0 for value in d):
            raise ValueError("Scientific result changed; revise plot domain after source audit")
        y=len(pairs)-1-i
        ax.plot([min(d),max(d)],[y,y],color="#a4b3bf",linewidth=2,zorder=1)
        col="#ad6047" if i<2 else "#277b86"
        ax.scatter(d,[y]*3,c=col,s=74,zorder=4,edgecolor="white",linewidth=1)
        m=statistics.mean(d)
        ax.plot([m,m],[y-.19,y+.19],c="#263e52",linewidth=2.3,zorder=3)
        ax.text(-42.3,y+.32,f"Mean {m:+.2f} pp",size=10,color="#344c60",ha="left")
    ax.axvline(0,c="#4d626c",linewidth=1.2,linestyle="--")
    ax.set_xlim(-43,3);ax.set_ylim(-.75,4)
    ax.set_yticks([3,2,1,0],[v[2] for v in pairs])
    ax.set_xlabel("Paired training-seed score difference (percentage points)")
    ax.set_title("NEU steel-defect GAN augmentation\nPaired effects across three training seeds",
                 loc="left",fontweight="semibold",fontsize=12,pad=25)
    ax.text(0,1.045,"Source: 12 executed original-data seed/arm results; same 360 real test photographs reused",
            transform=ax.transAxes,fontsize=9,color="#586d7d")
    ax.text(0,-.18,"One-sided exact sign test for 3/3 negative effects: p = 0.125. No cross-factory or cross-budget claim.",
            transform=ax.transAxes,fontsize=9,color="#586d7d")
    ax.grid(axis="x",alpha=.16)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    fig.subplots_adjust(left=.32,right=.97,bottom=.24,top=.82)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=250,facecolor="white")
    plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    render(args.csv,args.out)
