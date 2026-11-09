"""Generate measured Pass-12 figures 03, 20, 24 from real handwritten-digit images.

First run run_vae_evidence.py --out outputs/vae_audit to produce the checkpoint.
Noise in figures 20 and 24 is a controlled experimental intervention, not source data.
The denoiser is NOT a DDPM/DDIM diffusion model.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.nn import functional as F
from run_vae_evidence import VAE


def support(real, proposed, k=5):
    assert k < len(real) and k < len(proposed)
    real_nn = NearestNeighbors(n_neighbors=k+1).fit(real)
    prop_nn = NearestNeighbors(n_neighbors=k+1).fit(proposed)
    rr = real_nn.kneighbors(real)[0][:, -1]
    pr = prop_nn.kneighbors(proposed)[0][:, -1]
    a, ai = NearestNeighbors(n_neighbors=1).fit(real).kneighbors(proposed)
    b, bi = NearestNeighbors(n_neighbors=1).fit(proposed).kneighbors(real)
    return float(np.mean(a[:,0] <= rr[ai[:,0]])), float(np.mean(b[:,0] <= pr[bi[:,0]]))


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(65,128), nn.SiLU(), nn.Linear(128,128),
                                   nn.SiLU(), nn.Linear(128,64))

    def forward(self, x, sigma):
        return self.model(torch.cat([x, sigma], dim=1))


def run(out, epochs=35, seed=13):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    d = load_digits()
    X = d.data.astype("float32") / 16.0
    ix = np.arange(len(X))
    tr, va = train_test_split(ix, test_size=.25, random_state=seed, stratify=d.target)
    assert len(tr) == 1347 and len(va) == 450 and not set(tr).intersection(va)
    clean = X[va]
    checkpoint = out / "vae_model.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["seed"] == seed
    model = VAE()
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        reconstruction = model.decode(model.encode(torch.from_numpy(clean))[0]).numpy()
    scores = {"unmodified validation": support(X[tr], clean),
              "VAE reconstructed validation": support(X[tr], reconstruction)}
    fig, ax = plt.subplots(1,2,figsize=(8.4,3.3))
    for label, (precision, recall) in scores.items():
        ax[0].scatter(precision,recall,s=65)
        ax[0].annotate(label,(precision,recall),xytext=(6,6),textcoords="offset points",fontsize=8)
    ax[0].set(xlim=(0,1.04),ylim=(0,1.04),xlabel="5-NN support precision",ylabel="5-NN support recall")
    ax[1].imshow(np.concatenate([clean[0].reshape(8,8),reconstruction[0].reshape(8,8)],axis=1),
                 cmap="gray",vmin=0,vmax=1)
    ax[1].set_title("Held-out real digit | VAE reconstruction",fontsize=9)
    ax[1].axis("off")
    fig.tight_layout();fig.savefig(out/"figure_03.png",dpi=210);plt.close(fig)

    levels = [0.0,0.05,0.10,0.20,0.35]
    rng = np.random.default_rng(seed+20)
    robustness = []
    with torch.no_grad():
        for sigma in levels:
            corrupted = np.clip(clean+rng.normal(0,sigma,clean.shape).astype("float32"),0,1)
            pred = model.decode(model.encode(torch.from_numpy(corrupted))[0]).numpy()
            robustness.append({"sigma":sigma,"mae":float(np.abs(pred-clean).mean()),
                               "rmse":float(np.sqrt(((pred-clean)**2).mean()))})
    fig, ax = plt.subplots(1,2,figsize=(8.4,3.3))
    for key in ("mae","rmse"):
        ax[0].plot(levels,[r[key] for r in robustness],marker="o",label=key.upper())
    ax[0].set(xlabel="Applied Gaussian noise standard deviation",
              ylabel="Error vs clean real validation digits")
    ax[0].legend()
    ax[1].scatter(np.abs(reconstruction-clean).mean(axis=1),np.arange(len(clean)),s=4,alpha=.35)
    ax[1].set(xlabel="Per-digit reconstruction MAE",ylabel="Validation sample index")
    fig.tight_layout();fig.savefig(out/"figure_20.png",dpi=210);plt.close(fig)

    torch.manual_seed(seed+24)
    denoiser = Denoiser()
    optimizer = torch.optim.Adam(denoiser.parameters(),lr=.001)
    gen = torch.Generator().manual_seed(seed+240)
    training = torch.from_numpy(X[tr])
    history = []
    for _ in range(epochs):
        order = torch.randperm(len(training),generator=gen)
        denoiser.train()
        total = 0.0
        for ids in order.split(128):
            actual = training[ids]
            sigma = torch.rand((len(ids),1),generator=gen)*.45+.05
            noisy = torch.clamp(actual+sigma*torch.randn(actual.shape,generator=gen),0,1)
            optimizer.zero_grad()
            loss = F.mse_loss(denoiser(noisy,sigma),actual)
            loss.backward();optimizer.step()
            total += loss.detach().item()*len(ids)
        history.append(total/len(training))
    denoiser.eval()
    denoising = []
    levels24 = [.05,.15,.30,.45]
    rng = np.random.default_rng(seed+2400)
    fig, axes = plt.subplots(2,4,figsize=(8.4,4.1))
    with torch.no_grad():
        for j,sigma in enumerate(levels24):
            noisy = np.clip(clean+rng.normal(0,sigma,clean.shape).astype("float32"),0,1)
            pred = denoiser(torch.from_numpy(noisy),torch.full((len(clean),1),sigma)).numpy()
            row = {"sigma":sigma,"corrupted_mse":float(((noisy-clean)**2).mean()),
                   "denoised_mse":float(((pred-clean)**2).mean())}
            denoising.append(row)
            axes[0,j].imshow(noisy[0].reshape(8,8),cmap="gray",vmin=0,vmax=1)
            axes[1,j].imshow(np.clip(pred[0].reshape(8,8),0,1),cmap="gray",vmin=0,vmax=1)
            axes[0,j].set_title(f"Noisy real digit, sigma={sigma:.2f}",fontsize=8)
            axes[1,j].set_title(f"Denoised, MSE={row['denoised_mse']:.3f}",fontsize=8)
            axes[0,j].axis("off");axes[1,j].axis("off")
    fig.tight_layout();fig.savefig(out/"figure_24.png",dpi=210);plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.7,3.2))
    for key, label in (("corrupted_mse","Corrupted input"),("denoised_mse","Trained denoiser")):
        ax.plot(levels24,[r[key] for r in denoising],marker="o",label=label)
    ax.set(xlabel="Applied noise sigma",ylabel="Held-out real-digit MSE")
    ax.legend();fig.tight_layout()
    fig.savefig(out/"figure_24_metrics.png",dpi=210);plt.close(fig)

    result = {"dataset":"sklearn.datasets.load_digits real images",
              "training_count":len(tr),"validation_count":len(va),"test_count":0,
              "vae_checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "seed":seed,"precision_recall_5nn":scores,
              "vae_robustness":robustness,"denoiser_training_epochs":epochs,
              "denoiser_last_train_mse":history[-1],"denoising_validation":denoising,
              "limitations":"No independent test set. Added noise is controlled intervention. Denoiser is not DDPM or DDIM."}
    (out/"replacement_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"replacement_measurements.csv").open("w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["figure","condition","metric","value"])
        for label,(p,r) in scores.items():
            writer.writerow(["03",label,"support_precision_5nn",p])
            writer.writerow(["03",label,"support_recall_5nn",r])
        for row in robustness:
            for key in ("mae","rmse"):
                writer.writerow(["20",row["sigma"],key,row[key]])
        for row in denoising:
            for key in ("corrupted_mse","denoised_mse"):
                writer.writerow(["24",row["sigma"],key,row[key]])
    torch.save({"state_dict":denoiser.state_dict(),"seed":seed+24},out/"denoiser_model.pt")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("outputs/vae_audit"))
    p.add_argument("--denoise-epochs",type=int,default=35)
    args=p.parse_args()
    print(json.dumps(run(args.out,args.denoise_epochs),indent=2))
