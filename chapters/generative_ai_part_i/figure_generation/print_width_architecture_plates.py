"""Print-width diagrams of TWO actually implemented Chapter 6 VAE architectures.

Scientific conceptual figures from the checked-in model code, NOT observations
or newly generated chemistry/tissue; require separate real-result galleries.
"""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
NAVY="#173c56";TEAL="#127c84";COPPER="#aa704b";MUTED="#4f6879"
def stage(ax,x,y,w,h,name,detail,kind="model"):
    color=COPPER if kind=="latent" else TEAL if kind=="source" else NAVY
    fill="#fff6f0" if kind=="latent" else "#f0f7f7" if kind=="source" else "#f2f6f9"
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.011,rounding_size=.015",
      fc=fill,ec=color,lw=1.3)
    ax.add_patch(patch)
    ax.text(x+.017,y+h*.69,name,color=NAVY,fontsize=10.4,weight="bold",va="center")
    ax.text(x+.017,y+h*.29,detail,color=MUTED,fontsize=8.5,va="center")
def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,lw=1.45,color=MUTED))
def figure(out,name,title,subtitle,blocks,formula,footer):
    fig,ax=plt.subplots(figsize=(6.5,7.8),facecolor="white")
    ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
    fig.text(.055,.967,title,fontsize=14.4,color=NAVY,weight="bold")
    fig.text(.055,.929,subtitle,fontsize=8.8,color=MUTED)
    x,w,h=.065,.87,.104
    yvals=[.779,.636,.493,.350,.207]
    for i,(heading,detail,kind) in enumerate(blocks):
        stage(ax,x,yvals[i],w,h,heading,detail,kind)
        if i<len(blocks)-1:
            arrow(ax,(.5,yvals[i]-.005),(.5,yvals[i+1]+h+.005))
    fig.text(.055,.115,formula,fontsize=8.6,color=NAVY,weight="bold")
    fig.text(.055,.072,footer[0],fontsize=8.2,color=MUTED)
    fig.text(.055,.046,footer[1],fontsize=8.2,color=MUTED)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"{name}.{ext}",dpi=245,facecolor="white")
    plt.close(fig)
def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    figure(out,"smiles_vae_architecture_compact",
       "NCI SMILES VAE · implemented model",
       "Source-faithful character-sequence architecture (not chemical experimental results)",
       [("Original NCI input strings","600 canonical source SMILES; 360 / 120 / 120 train / val / test","source"),
        ("32-feature character embedding","Bidirectional GRU encoder, 128 hidden units per direction","model"),
        ("Gaussian latent representation","Separate μ and log variance heads → 32-dimensional z","latent"),
        ("Autoregressive GRU decoder","128 hidden features; 32-d z appended to each token embedding","model"),
        ("Actual two-mode evaluation","120 greedy test reconstructions; 1,024 sampled prior strings","source")],
       "Training: next-token cross-entropy + scheduled β × KL / 32.",
       ("RDKit verifies validity, canonical identity and Morgan similarity.",
        "A valid SMILES is not an experimentally synthesized or effective molecule."))
    figure(out,"tissue_vae_architecture_compact",
       "Histology VAE · implemented model",
       "Source-faithful convolutional encoder and decoder (not patient-level evidence)",
       [("Authentic photographed tissue input","One original 512 × 512 photo; actual 32 × 32 RGB patches","source"),
        ("Three stride-2 encoder convolutions","Channel sequence 3 → 16 → 32 → 64; feature map 64 × 4 × 4","model"),
        ("Gaussian 24-dimensional bottleneck","Separate learned μ, log variance; stochastic z sampling","latent"),
        ("Linear projection and decoder","24 → 64 × 4 × 4; transpose convolutions 64 → 32 → 16 → 3","model"),
        ("Real held-out patch reconstruction","589 training / 93 validation / 217 test; sigmoid RGB outputs","source")],
       "Training: pixel mean squared error + 0.0001 × KL.",
       ("217 spatial test patches originate from the SAME original tissue photograph.",
        "Prior outputs are generated images, not medical diagnosis or clinical data."))
    (out/"architecture_scope.json").write_text(json.dumps({
       "scientific_figures":"code-faithful conceptual, not measured result figures",
       "source_models":["molecular SMILES bidirectional GRU VAE","tissue convolutional VAE"],
       "figure_width_inches":6.5,
       "production_approved":False},indent=2)+"\n")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args();main(a.output)
