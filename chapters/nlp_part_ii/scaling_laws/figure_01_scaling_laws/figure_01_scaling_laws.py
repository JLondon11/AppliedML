"""NLP Part II Figure 1 — scaling laws.
Published fits only; no fabricated empirical observations.
Kaplan et al. (2020): alpha_N=.076, alpha_D=.103, Nc=6.4e13, Dc=1.8e13.
Hoffmann et al. (2022): E=1.69, A=406.4, B=410.7, alpha=.34, beta=.28.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"]="none"
OUT=Path(__file__).resolve().parent
alpha_N,alpha_D=.076,.103; Nc,Dc=6.4e13,1.8e13
E,A,B=1.69,406.4,410.7; alpha,beta=.34,.28
a=beta/(alpha+beta); b=alpha/(alpha+beta)
G=((alpha*A)/(beta*B))**(1/(alpha+beta))
N=np.logspace(6,11.5,240); D=np.logspace(7,12.5,240); C=np.logspace(18,25.5,280)
LN=(Nc/N)**alpha_N; LD=(Dc/D)**alpha_D
Nopt=G*(C/6)**a; Dopt=(1/G)*(C/6)**b
LC=E+A/Nopt**alpha+B/Dopt**beta
pd.DataFrame({"parameters_N":N,"kaplan_L_N_infinite_data":LN}).to_csv(OUT/"figure_01_panel_a_kaplan_model_size.csv",index=False)
pd.DataFrame({"tokens_D":D,"kaplan_L_D_infinite_model":LD}).to_csv(OUT/"figure_01_panel_b_kaplan_data_size.csv",index=False)
pd.DataFrame({"compute_C_flops":C,"hoffmann_N_opt":Nopt,"hoffmann_D_opt":Dopt,"hoffmann_compute_optimal_loss":LC}).to_csv(OUT/"figure_01_panel_c_hoffmann_compute_optimal.csv",index=False)
fig,ax=plt.subplots(1,3,figsize=(13.2,4.3))
ax[0].loglog(N,LN,lw=1.8); ax[0].set(xlabel="Non-embedding parameters, $N$",ylabel="Cross-entropy loss"); ax[0].text(.06,.10,r"$L(N,\infty)=(N_c/N)^{0.076}$",transform=ax[0].transAxes,fontsize=9)
ax[1].loglog(D,LD,lw=1.8); ax[1].set(xlabel="Training tokens, $D$",ylabel="Cross-entropy loss"); ax[1].text(.06,.10,r"$L(\infty,D)=(D_c/D)^{0.103}$",transform=ax[1].transAxes,fontsize=9)
ax[2].semilogx(C,LC,lw=1.8); ax[2].set(xlabel="Training compute, $C$ (FLOPs)",ylabel="Compute-optimal loss"); ax[2].text(.06,.10,r"$\hat L=1.69+406.4N^{-0.34}+410.7D^{-0.28}$",transform=ax[2].transAxes,fontsize=8.5)
for lab,a0 in zip(["(a)","(b)","(c)"],ax): a0.text(.5,-.20,lab,transform=a0.transAxes,ha="center",va="top",fontsize=12); a0.tick_params(direction="out"); a0.grid(False)
fig.tight_layout(w_pad=2.2); fig.savefig(OUT/"figure_01_scaling_laws.svg",bbox_inches="tight"); fig.savefig(OUT/"figure_01_scaling_laws.png",dpi=300,bbox_inches="tight")
