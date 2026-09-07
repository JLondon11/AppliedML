from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"]="none"
OUT=Path(__file__).resolve().parent
E,A,B=1.69,406.4,410.7; alpha,beta=.34,.28
a=beta/(alpha+beta); b=alpha/(alpha+beta)
G=((alpha*A)/(beta*B))**(1/(alpha+beta))
loss=lambda N,D:E+A/N**alpha+B/D**beta
optimal=lambda C:(G*(C/6)**a,(1/G)*(C/6)**b)
C0=5e23; N0,D0=optimal(C0)
r=np.logspace(-1.2,1.2,321); N=N0*r; D=C0/(6*N); L=loss(N,D)
C=np.logspace(20,25.5,240); No,Do=optimal(C); Lo=loss(No,Do)
pd.DataFrame({"compute_flops":C0,"N_over_Nopt":r,"parameters_N":N,"training_tokens_D":D,"predicted_loss":L}).to_csv(OUT/"figure_02_fixed_compute_isoflop.csv",index=False)
pd.DataFrame({"compute_flops":C,"N_opt":No,"D_opt":Do,"predicted_optimal_loss":Lo}).to_csv(OUT/"figure_02_compute_optimal_frontier.csv",index=False)
fig,ax=plt.subplots(1,3,figsize=(13.4,4.35))
i0=np.argmin(abs(r-1)); ip=np.argmin(abs(r-4)); idat=np.argmin(abs(r-.25))
ax[0].loglog(N,D,lw=1.6)
for i,m in [(i0,"o"),(ip,"s"),(idat,"^")]: ax[0].scatter([N[i]],[D[i]],s=46,marker=m)
ax[0].set(xlabel="Parameters, $N$",ylabel="Training tokens, $D$")
ax[1].semilogx(r,L,lw=1.7)
for i,m in [(i0,"o"),(ip,"s"),(idat,"^")]: ax[1].scatter([r[i]],[L[i]],s=46,marker=m)
ax[1].axvline(1,lw=.8,ls="--"); ax[1].set(xlabel="$N/N_{\\mathrm{opt}}$ at fixed compute",ylabel="Predicted loss")
ax[2].loglog(C,No,lw=1.7,label="$N_{\\mathrm{opt}}$"); ax[2].loglog(C,Do,lw=1.7,label="$D_{\\mathrm{opt}}$")
ax[2].set(xlabel="Training compute, $C$ (FLOPs)",ylabel="Optimal scale"); ax[2].legend(frameon=False)
for lab,a0 in zip(["(a)","(b)","(c)"],ax): a0.text(.5,-.20,lab,transform=a0.transAxes,ha="center",va="top"); a0.grid(False)
fig.tight_layout(); fig.savefig(OUT/"figure_02_compute_optimal_scaling.svg",bbox_inches="tight"); fig.savefig(OUT/"figure_02_compute_optimal_scaling.png",dpi=300,bbox_inches="tight")
