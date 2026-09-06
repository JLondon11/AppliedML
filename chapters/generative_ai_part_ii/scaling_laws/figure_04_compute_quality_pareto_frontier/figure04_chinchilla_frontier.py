import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

E, A, B = 1.69, 406.4, 410.7
alpha, beta = 0.34, 0.28

def loss(N, D):
    return E + A / N**alpha + B / D**beta

def optimal(C):
    N = ((alpha*A)/(beta*B))**(1/(alpha+beta)) * (C/6)**(beta/(alpha+beta))
    D = C/(6*N)
    return N, D

out = Path(__file__).resolve().parent
C = np.logspace(18.8, 25.5, 240)
N, D = optimal(C)
L = loss(N, D)
N_small, N_large = N/4, 4*N
L_small = loss(N_small, C/(6*N_small))
L_large = loss(N_large, C/(6*N_large))

pd.DataFrame({
    "compute_flops": C,
    "N_opt_parameters": N,
    "D_opt_tokens": D,
    "predicted_loss_optimal": L,
    "predicted_loss_Nopt_div4": L_small,
    "predicted_loss_4x_Nopt": L_large,
}).to_csv(out/"figure04_chinchilla_frontier_data.csv", index=False)

fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=220)
ax.plot(C, L, lw=2.4, color="#355C7D", label="Compute-optimal allocation")
ax.plot(C, L_small, lw=1.35, ls="--", color="#7A8FA6", label=r"$N=N_{\rm opt}/4$")
ax.plot(C, L_large, lw=1.35, ls="-.", color="#A65F2B", label=r"$N=4N_{\rm opt}$")
ax.set_xscale("log")
ax.set_xlabel("Training compute, $C$ (FLOPs)")
ax.set_ylabel(r"Predicted validation loss, $\hat{L}$")
ax.grid(True, which="major", lw=0.55, color="#D8DADD")
ax.grid(False, which="minor")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(out/"genai2_fig04_compute_quality_pareto_frontier.png", bbox_inches="tight", facecolor="white")
fig.savefig(out/"genai2_fig04_compute_quality_pareto_frontier.svg", bbox_inches="tight", facecolor="white")
