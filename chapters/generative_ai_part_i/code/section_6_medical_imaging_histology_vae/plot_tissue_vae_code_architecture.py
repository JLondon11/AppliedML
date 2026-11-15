"""Exact-model scientific architecture schematic of the original tissue-patch VAE.

A conceptual model illustration, not actual generated tissue, measured clinical
outputs or a replacement for source/model-output scientific image galleries.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
NAVY="#183F5A";BLUE="#36758A";TEAL="#137D83";AMBER="#B76C41";GRAY="#4D687B"

def block(ax,x,y,w,h,heading,detail,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=0.021",
  ec=color,lw=1.45,fc=("#F0F6F8" if color!=AMBER else "#FFF5EF"),zorder=3))
 ax.text(x+w/2,y+h*.65,heading,ha="center",va="center",fontsize=11.2,color=NAVY,weight="bold")
 ax.text(x+w/2,y+h*.29,detail,ha="center",va="center",fontsize=8.7,color=GRAY)
def link(ax,a,b):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,
    linewidth=1.6,color="#657F8D",zorder=1))

def main(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 fig,ax=plt.subplots(figsize=(15,8.45),facecolor="white")
 ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
 fig.text(.04,.947,"Convolutional tissue VAE · exact implemented encoder and decoder",
  fontsize=18.5,color=NAVY,weight="bold")
 fig.text(.04,.912,"32 × 32 RGB patches from one genuine microscope photo; scientific architecture diagram, not generated microscopy data.",
  fontsize=10,color=GRAY)
 ax.text(.044,.822,"A  ENCODING ORIGINAL IMAGE PATCHES",color=NAVY,fontsize=11.2,weight="bold")
 block(ax,.04,.617,.155,.125,"Real 32 × 32","RGB tissue patch",TEAL)
 block(ax,.237,.617,.190,.125,"Conv encoder","3→16→32→64 channels")
 block(ax,.475,.617,.181,.125,"Spatial features","64 × 4 × 4 flattened")
 block(ax,.708,.617,.239,.125,"Latent distribution","μ, log σ² → 24 components",AMBER)
 for a,b in ((.195,.237),(.427,.475),(.656,.708)):link(ax,(a,.680),(b,.680))
 ax.text(.50,.555,"Reparameterization: z = μ + exp(½ log σ²) · ε,     ε ∼ N(0, I)",
  ha="center",color=NAVY,fontsize=10.7)
 ax.text(.044,.480,"B  DECODING AND EVALUATING PATCHES",color=NAVY,fontsize=11.2,weight="bold")
 block(ax,.05,.283,.225,.125,"Latent code z ∈ ℝ²⁴","posterior μ or normal prior",AMBER)
 block(ax,.325,.283,.238,.125,"Linear + reshape","24 → 64 × 4 × 4")
 block(ax,.611,.283,.332,.125,"Transposed-convolution decoder","64→32→16→3; sigmoid RGB")
 for a,b in ((.275,.325),(.563,.611)):link(ax,(a,.344),(b,.344))
 ax.add_patch(FancyBboxPatch((.035,.102),.925,.108,
  boxstyle="round,pad=.012,rounding_size=.02",lw=1,ec="#d6e1e6",fc="#f5f8fa"))
 ax.text(.059,.177,"Original training loss: pixel MSE + 10⁻⁴ × KL  (KL summed over 24 latent features and averaged by sample)",
  color=NAVY,fontsize=10.4,weight="bold")
 ax.text(.059,.135,"Evaluation: 589 source-region training patches; 93 validation; 217 held-out patches from different rows of ONE original photo.",
  color=GRAY,fontsize=9.65)
 fig.text(.04,.054,"The true model-output gallery separately shows measured reconstructions and generated prior samples; this schematic is not a prediction.",
  fontsize=9.1,color=GRAY)
 for ext in ("pdf","svg","png"):
  fig.savefig(output/f"tissue_convolutional_vae_code_architecture.{ext}",dpi=220,facecolor="white")
 plt.close(fig)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path)
 args=p.parse_args();main(args.output)
