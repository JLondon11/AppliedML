"""Publication-quality scientific schematic of the ACTUAL NCI character-level VAE.

The diagram traces topology and evaluated outputs read from run.py. It is an
architecture explanation, not a model output, new data or fabricated results.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
INK="#173B55";SECOND="#45647A";TEAL="#087D82";BLUE="#2A6B89";AMBER="#B26C41";LIGHT="#F1F6F8"
def box(ax,xy,w,h,title,subtitle="",edge=BLUE,face=LIGHT):
    x,y=xy
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.014,rounding_size=0.018",
       linewidth=1.5,edgecolor=edge,facecolor=face,zorder=2))
    ax.text(x+w/2,y+h*.65,title,ha="center",va="center",color=INK,fontsize=10,weight="bold",zorder=3)
    if subtitle:ax.text(x+w/2,y+h*.28,subtitle,ha="center",va="center",
        color=SECOND,fontsize=8.2,zorder=3)
def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,linewidth=1.5,
       color=SECOND,connectionstyle="arc3,rad=0",zorder=1))
def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(15.4,8.9),facecolor="white")
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    fig.text(.045,.949,"Character-level molecular VAE · architecture and genuine-data evaluation",
        fontsize=19,color=INK,weight="bold")
    fig.text(.045,.913,"Schematic transcribed from the executed Chapter 6 NCI model; arrows show implemented tensor flow, not observed performance.",
        fontsize=10,color=SECOND)
    ax.add_patch(FancyBboxPatch((.025,.505),.95,.34,boxstyle="round,pad=0.015,rounding_size=0.013",
       edgecolor="#CFDDE4",linewidth=1.2,facecolor="#FBFCFD",zorder=0))
    ax.text(.044,.818,"A  MODEL ARCHITECTURE  —  600 original NCI structures (360 / 120 / 120 split)",
       color=INK,fontsize=11,weight="bold")
    box(ax,(.045,.632),.139,.117,"Original NCI SMILES","real chemical strings",edge=TEAL)
    box(ax,(.211,.632),.137,.117,"Embedding","32 features / character")
    box(ax,(.377,.632),.135,.117,"BiGRU encoder","hidden 128 × 2")
    box(ax,(.545,.632),.138,.117,"μ and log σ²","two learned heads → 32")
    box(ax,(.722,.632),.208,.117,"Autoregressive GRU decoder",
       "hidden 128; z appended each token",edge=AMBER,face="#FFF7F0")
    for a,b in ((.184,.211),(.348,.377),(.512,.545),(.683,.722)):
        arrow(ax,(a,.691),(b,.691))
    ax.text(.50,.594,"Reparameterization: z = μ + exp(½ log σ²) · ε,   ε ~ N(0, I)",
       color=INK,fontsize=10,ha="center")
    ax.text(.50,.536,"Training: token cross-entropy (non-padding)  +  β(epoch) · KL / 32",
       ha="center",fontsize=10.5,color=BLUE,weight="bold")
    ax.text(.50,.460,"B  TWO EVALUATION MODES — outputs passed to RDKit checks",
        color=INK,fontsize=11,ha="center",weight="bold")
    box(ax,(.053,.292),.256,.135,"Held-out test decoding",
        "120 original source molecules; greedy reconstruction",edge=TEAL)
    box(ax,(.373,.292),.254,.135,"New prior samples",
        "1,024 strings from z ~ N(0, I)",edge=AMBER,face="#FFF7F0")
    box(ax,(.690,.292),.249,.135,"RDKit chemistry checks",
        "validity · canonical identity · Morgan similarity",edge=BLUE)
    ax.add_patch(FancyBboxPatch((.025,.102),.95,.139,boxstyle="round,pad=0.012,rounding_size=0.014",
       edgecolor="#DAE4E8",facecolor="#F7FAFB",zorder=0))
    ax.text(.047,.201,"C  ORIGINAL EXPERIMENT INTERPRETATION",
       fontsize=11,color=INK,weight="bold")
    ax.text(.048,.162,"• The encoder/decoder reconstructs SMILES strings, not measured biological activity. Training uses six epochs and a validation-selected checkpoint.",
       fontsize=9.6,color=SECOND)
    ax.text(.048,.129,"• Prior samples are model-generated strings; RDKit-valid does not imply synthesizable, safe, novel, or biologically active molecules.",
       fontsize=9.6,color=SECOND)
    fig.text(.045,.058,"Scientific source: code/section_6_molecular_smiles_vae/run.py  ·  original RDKit NCI first_5K.smi  ·  architecture diagram, not new inference data",
       fontsize=9,color=SECOND)
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"nci_molecular_vae_code_architecture.{ext}",dpi=240,facecolor="white")
    plt.close(fig)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.output)
