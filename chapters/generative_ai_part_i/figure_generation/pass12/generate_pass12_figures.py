"""Run all six Pass-12 figures exclusively from real handwritten-digit experiments.

Figures 03, 10, 11, 14, 20, 24 are measured from sklearn digits.
Figure 20 applies controlled perturbations to real validation data.
Figure 24 uses a genuinely trained DDPM with held-out test metrics.
DDIM and score-based SDE results are NOT claimed.
"""
import csv
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_vae_evidence import run as run_vae
from real_data_figures import run as run_replacements
from ddpm_digits_experiment import run as run_ddpm
from compose_ddpm_figure24 import render as render_ddpm_figure

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def render_evidence_outputs():
    OUT.mkdir(parents=True, exist_ok=True)
    # Genuine trained-model checkpoint, measured epoch histories and validation embeddings.
    run_vae(OUT, epochs=60, seed=13)
    run_replacements(OUT, epochs=35, seed=13)
    ddpm_out=OUT/'ddpm'
    run_ddpm(ddpm_out, epochs=45, seed=17)
    # Override the older conditional-denoiser graphic with authentic DDPM samples.
    render_ddpm_figure(ddpm_out, OUT/'figure_24.png')
    shutil.copyfile(OUT / "vae_interpolation.png", OUT / "figure_10.png")

    with (OUT / "history.csv").open(newline="") as f:
        history = list(csv.DictReader(f))
    epoch = np.asarray([int(x["epoch"]) for x in history])
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
    for column,label in (("train_elbo","Training negative ELBO"),
                         ("val_elbo","Validation negative ELBO")):
        axes[0].plot(epoch, [float(x[column]) for x in history], label=label)
    axes[0].set(xlabel="Epoch",ylabel="Negative ELBO / sample")
    axes[0].legend(frameon=False,fontsize=7)
    for column,label in (("train_bce","Reconstruction BCE"),("train_kl","KL divergence")):
        axes[1].plot(epoch, [float(x[column]) for x in history], label=label)
    axes[1].set(xlabel="Epoch",ylabel="Training loss component / sample")
    axes[1].legend(frameon=False,fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT/"figure_11.png",dpi=220)
    plt.close(fig)

    with (OUT/"validation_embeddings.csv").open(newline="") as f:
        points = list(csv.DictReader(f))
    Z = np.array([[float(x["latent_mu_1"]),float(x["latent_mu_2"])] for x in points])
    labels = np.array([int(x["true_digit"]) for x in points])
    purity = np.array([float(x["knn_label_purity"]) for x in points])
    fig, axes = plt.subplots(1,2,figsize=(7.8,3.0))
    axes[0].scatter(Z[:,0],Z[:,1],c=labels,cmap="tab10",s=9,alpha=.8,linewidths=0)
    axes[0].set(xlabel="Latent mean 1",ylabel="Latent mean 2")
    axes[1].hist(purity,bins=np.linspace(0,1,11))
    axes[1].axvline(purity.mean(),ls="--",lw=1)
    axes[1].set(xlabel="10-NN validation label purity",ylabel="Validation images")
    fig.tight_layout()
    fig.savefig(OUT/"figure_14.png",dpi=220)
    plt.close(fig)
    for number in (3,10,11,14,20,24):
        assert (OUT / f"figure_{number:02d}.png").stat().st_size > 0


if __name__ == "__main__":
    render_evidence_outputs()
