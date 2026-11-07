"""Single-step diffusion epsilon-prediction on genuine scikit-image photographs.

This is a diagnostic denoiser, not a DDPM sampler or a generative benchmark.
Run: python chapters/generative_ai_part_i/code/section_8_12_1_image_restoration/run.py
Requires numpy, scipy, scikit-image, torch, matplotlib.
"""
import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import color, data, metrics, transform
import skimage
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 20261106
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

def gray(im):
    if im.ndim == 3:
        im = color.rgb2gray(im[:, :, :3])
    if np.issubdtype(im.dtype, np.integer):
        im = im.astype(np.float32) / np.iinfo(im.dtype).max
    return transform.resize(np.asarray(im, dtype=np.float32), (128, 128),
                            anti_aliasing=True).astype(np.float32)

train = {k: gray(getattr(data, k)()) for k in ("astronaut", "coffee", "chelsea", "rocket")}
test = {k: gray(getattr(data, k)()) for k in ("camera", "coins")}

class EpsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 1, 3, padding=1))
    def forward(self, noisy, alpha):
        return self.net(torch.cat([noisy, alpha[:, None, None, None].expand_as(noisy)], dim=1))

model = EpsCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
images = np.stack(list(train.values()))
rng = np.random.default_rng(SEED)
losses = []
for step in range(420):
    indices = rng.integers(0, len(images), size=32)
    xs = rng.integers(0, 97, size=32)
    ys = rng.integers(0, 97, size=32)
    clean = np.stack([images[k, y:y+32, x:x+32] for k, x, y in zip(indices, xs, ys)])[:, None]
    clean = torch.from_numpy(clean.copy())
    eps = torch.randn_like(clean)
    alpha = torch.from_numpy(rng.uniform(0.70, 0.98, size=32).astype(np.float32))
    noisy = alpha[:, None, None, None].sqrt() * clean + (1-alpha[:, None, None, None]).sqrt() * eps
    loss = ((model(noisy, alpha) - eps)**2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

rows = []
panels = {}
model.eval()
for name, image in test.items():
    for alpha_bar in (0.90, 0.75):
        noise_rng = np.random.default_rng(SEED + len(name)*100 + int(alpha_bar*100))
        eps = noise_rng.standard_normal(image.shape).astype(np.float32)
        noisy = (np.sqrt(alpha_bar)*image + np.sqrt(1-alpha_bar)*eps).astype(np.float32)
        with torch.no_grad():
            prediction = model(torch.from_numpy(noisy[None,None]),
                               torch.tensor([alpha_bar], dtype=torch.float32)).numpy()[0,0]
        denoised = np.clip((noisy - np.sqrt(1-alpha_bar)*prediction)/np.sqrt(alpha_bar), 0, 1)
        input_estimate = np.clip(noisy/np.sqrt(alpha_bar), 0, 1)
        baseline = np.clip(gaussian_filter(input_estimate, sigma=0.8), 0, 1)
        for method, output in (("noisy", input_estimate), ("gaussian_0.8", baseline),
                               ("diffusion_eps_CNN", denoised)):
            rows.append({"image": name, "alpha_bar": alpha_bar, "method": method,
                         "PSNR_dB": round(float(metrics.peak_signal_noise_ratio(image, output, data_range=1)), 4),
                         "SSIM": round(float(metrics.structural_similarity(image, output, data_range=1)), 5)})
        panels[(name,alpha_bar)] = (image, input_estimate, baseline, denoised)

with (OUT/"metrics.csv").open("w",newline="") as file:
    writer = csv.DictWriter(file,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (OUT/"training_loss.csv").open("w",newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["step","mse_epsilon"])
    writer.writerows(enumerate(losses,1))
fig, axes = plt.subplots(2,4,figsize=(12,6.2),constrained_layout=True)
for i,(name,alpha_bar) in enumerate((("camera",0.75),("coins",0.75))):
    for j,(im,title) in enumerate(zip(panels[(name,alpha_bar)],
                        ("Reference","Corrupted input","Gaussian baseline","Diffusion noise predictor"))):
        axes[i,j].imshow(im,cmap="gray",vmin=0,vmax=1)
        axes[i,j].axis("off")
        if i==0: axes[i,j].set_title(title,fontsize=10)
        if j==0:
            axes[i,j].text(0.02,0.96,name,ha="left",va="top",color="white",
                fontsize=10,transform=axes[i,j].transAxes,
                bbox={"facecolor":"black","alpha":0.5,"pad":2})
fig.savefig(FIG/"real_photo_diffusion_restoration.png",dpi=200,bbox_inches="tight")
plt.close(fig)
fig,ax=plt.subplots(figsize=(7.4,3.6))
ax.plot(np.arange(1,len(losses)+1),losses,lw=0.7,alpha=0.55)
window=15
ax.plot(np.arange(window,len(losses)+1),
        np.convolve(losses,np.ones(window)/window,mode="valid"),
        lw=1.8,label="15-step moving average")
ax.set(xlabel="Optimizer step",ylabel="Epsilon prediction MSE")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG/"real_photo_diffusion_training.png",dpi=200)
plt.close(fig)
manifest={"experiment":"time-conditioned epsilon prediction, single-step restoration, not DDPM sampling",
          "seed":SEED,"steps":420,"batch":32,"patch":32,
          "training_images":list(train),"heldout_test_images":list(test),
          "training_source":"scikit-image data photographs",
          "data_split":"four separate training photographs; two separate test photographs",
          "alpha_bar":[0.90,0.75],"torch_version":torch.__version__,
          "skimage_version":skimage.__version__,
          "baseline":"fixed Gaussian sigma 0.8, not validation-tuned",
          "limitations":"No independent validation set; only two holdout images; no full diffusion sampler",
          "image_sha256":{name:hashlib.sha256(im.tobytes()).hexdigest()
                          for name,im in {**train,**test}.items()}}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
torch.save({"state_dict":model.state_dict(),"manifest":manifest},OUT/"checkpoint.pt")
print(json.dumps(rows,indent=2))
