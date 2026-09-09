"""Reproduce Scientific AI Figure 7; no false cross-cohort patient pairing."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
fig,ax=plt.subplots(figsize=(15.8,7)); ax.set(xlim=(0,16),ylim=(0,7.1)); ax.axis("off")
ink="#30343A"; line="#747A7F"; pale="#F4F5F5"
def box(x,y,w,h,label,edge="#536B7A",sub=""):
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor="white",edgecolor=edge,lw=.9,zorder=3)); ax.text(x,y+.08 if sub else y,label,ha="center",va="center",fontsize=8.7)
    if sub: ax.text(x,y-h*.3,sub,ha="center",va="center",fontsize=6.8,color="#676C71")
for x,y,label,sub in [(1.65,5.75,"CBIS-DDSM / INbreast","mammography"),(1.65,4.35,"breast MRI cohort","MRI"),(1.65,2.95,"TCGA-BRCA","histopathology"),(1.65,1.55,"TCGA-BRCA","genomics / molecular")]: box(x,y,2.4,.76,label,sub=sub)
for x,y,l in [(5.7,5.75,r"$E_{mam}$"),(5.7,4.35,r"$E_{MRI}$"),(5.7,2.95,r"$E_{path}$"),(5.7,1.55,r"$E_{gen}$")]:
    box(x,y,1.35,.66,l,sub="encoder")
for y in [5.75,4.35,2.95,1.55]: ax.add_patch(FancyArrowPatch((2.9,y),(5.0,y),arrowstyle="-|>",mutation_scale=9,lw=.8,color=line))
box(9.05,4.55,2.65,1.0,"Paired patient-level cohort",edge=ink,sub="institutional / prospective")
for y1,y2 in [(5.75,4.85),(4.35,4.65),(2.95,4.45),(1.55,4.25)]: ax.add_patch(FancyArrowPatch((6.4,y1),(7.7,y2),arrowstyle="-|>",mutation_scale=8,lw=.7,color=line))
box(11.65,4.55,1.7,.82,r"$\mathcal{F}(h_1,\ldots,h_M)$",sub="multimodal fusion"); box(14,4.55,2.15,.82,r"$z_{patient}$",sub="patient-level state")
ax.add_patch(FancyArrowPatch((10.4,4.55),(10.78,4.55),arrowstyle="-|>",mutation_scale=9,lw=.8,color=line)); ax.add_patch(FancyArrowPatch((12.5,4.55),(12.9,4.55),arrowstyle="-|>",mutation_scale=9,lw=.8,color=line))
ax.text(9.05,3.72,"same patient: imaging + pathology + molecular assays",ha="center",fontsize=7.4)
ax.text(4,6.55,"Public resources support modality-specific pretraining / benchmarking;\nthey must not be treated as if they were the same patients.",ha="center",va="top",fontsize=8.2)
ax.text(11.7,1.5,"Fusion validity requires patient-level identity, acquisition provenance,\nmissing-modality handling, calibration, and external validation.",ha="center",fontsize=8.1)
fig.savefig("../figures/figure_07_multimodal_precision_oncology.svg",bbox_inches="tight",facecolor="white")
