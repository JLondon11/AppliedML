from pathlib import Path
import argparse, json
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    rows=[
      {"method":"BEVFormer","modality":"Camera","nds":56.9},
      {"method":"CenterPoint","modality":"LiDAR","nds":65.5},
      {"method":"BEVFormer-M","modality":"Fusion","nds":72.9},
    ]
    df=pd.DataFrame(rows)
    df.to_csv(out/"ch01_autonomous_driving_benchmark.csv",index=False)

    colors={"Camera":"#355C7D","LiDAR":"#4F7C6E","Fusion":"#B26E3B"}
    fig,ax=plt.subplots(figsize=(8.4,3.6))
    bars=ax.bar(df["method"],df["nds"],color=[colors[m] for m in df["modality"]],width=.62)
    ax.set_ylabel("nuScenes Detection Score (NDS)")
    ax.set_ylim(45,80)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(direction="out")
    for bar,row in zip(bars,rows):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.8,
                f'{row["nds"]:.1f}\n{row["modality"]}',ha="center",va="bottom",fontsize=8)
    fig.tight_layout(pad=.25)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"ch01_autonomous_driving_nds.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)

    meta={
      "source":"Pass20 published benchmark anchors for nuScenes detection",
      "warning":"Methods use different sensor modalities; values are not a modality-controlled ranking.",
      "values":rows
    }
    (out/"ch01_autonomous_driving_provenance.json").write_text(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
