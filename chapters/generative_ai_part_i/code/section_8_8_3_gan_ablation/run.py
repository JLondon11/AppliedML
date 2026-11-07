"""Real-photo GAN augmentation ablation; not an annotated industrial-defect benchmark."""
import argparse, csv, hashlib, json, random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
import torch
from torch import nn
from texture_gan_reference import Generator, Discriminator, generate_patches, texture_features, dist

SEED = 20261106

def train_one(name, train_patches, heldout, augment, steps, out):
    torch.manual_seed(SEED)
    gen, disc = Generator(32), Discriminator()
    og = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    source = torch.from_numpy(train_patches[:, None].copy())
    history = []
    gen.train()
    for step in range(1, steps+1):
        indices = torch.randint(len(source), (32,))
        real = source[indices].clone()
        if augment:
            flags = torch.rand(32) < 0.5
            real[flags] = real[flags].flip(-1)
        fake = gen(torch.randn(32, 32))
        dloss = (criterion(disc(real), torch.ones(32, 1)) +
                 criterion(disc(fake.detach()), torch.zeros(32, 1))) / 2
        od.zero_grad(); dloss.backward(); od.step()
        fake = gen(torch.randn(32, 32))
        gloss = criterion(disc(fake), torch.ones(32, 1))
        og.zero_grad(); gloss.backward(); og.step()
        history.append((name, int(augment), step, float(dloss.detach()), float(gloss.detach())))
    gen.eval()
    torch.manual_seed(SEED+100)
    with torch.no_grad():
        generated = gen(torch.randn(100, 32)).numpy()[:, 0]
    torch.save({"generator": gen.state_dict(), "steps":steps,
                "augmentation":augment, "seed":SEED},
               out/f'generator_{name}_{"flip" if augment else "none"}.pt')
    scale = np.maximum(texture_features(train_patches).std(0), 0.05)
    score = dist(generated, heldout, scale)
    np.save(out/f'generated_{name}_{"flip" if augment else "none"}.npy', generated)
    return float(score), generated, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"results")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    figures = args.output/"figures"; figures.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    random.seed(SEED); np.random.seed(SEED)
    rows, hist, results, hashes = [], [], [], {}
    for name in ("brick","gravel","grass"):
        photo = getattr(data,name)()
        hashes[name] = hashlib.sha256(photo.tobytes()).hexdigest()
        train = generate_patches(photo,0,384)
        heldout = generate_patches(photo,384,512)
        if len(train)!=713 or len(heldout)!=217:
            raise ValueError(f"Unexpected patch split: {name}")
        for flag in (False,True):
            score, samples, history = train_one(name,train,heldout,flag,args.steps,args.output)
            rows.append({"texture":name,"augment_horizontal_flip":int(flag),"steps":args.steps,
                         "n_training_patches":len(train),"n_heldout_patches":len(heldout),
                         "n_generated":len(samples),"feature_distance":round(score,6)})
            hist.extend(history); results.append((name,flag,samples))
    with (args.output/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (args.output/"losses.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["texture","augmented","step","D_loss","G_loss"]);w.writerows(hist)
    fig,ax=plt.subplots(figsize=(7,3.7))
    labels=["brick","gravel","grass"]
    x=np.arange(3)
    for k,aug in enumerate((0,1)):
        values=[next(r["feature_distance"] for r in rows if r["texture"]==name
                     and r["augment_horizontal_flip"]==aug) for name in labels]
        ax.bar(x+(k-0.5)*0.33,values,width=0.32,
               label="Horizontal flips" if aug else "No augmentation")
    ax.set_xticks(x,labels);ax.set_ylabel("Standardized mean texture-feature distance (lower better)")
    ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(figures/"augmentation_ablation_metrics.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,6,figsize=(10,5),constrained_layout=True)
    for i,name in enumerate(labels):
        for j,flag in enumerate((False,True)):
            patch=next(im for n,a,im in results if n==name and a==flag)
            for k in range(3):
                axes[i,j*3+k].imshow((patch[k]+1)/2,vmin=0,vmax=1,cmap="gray")
                axes[i,j*3+k].axis("off")
                if i==0 and k==1: axes[i,j*3+k].set_title("Flip" if flag else "No flip")
                if j==0 and k==0: axes[i,0].set_ylabel(name)
    fig.savefig(figures/"augmentation_ablation_samples.png",dpi=180);plt.close(fig)
    manifest={"dataset":"scikit-image brick/gravel/grass photographs","source_hashes":hashes,
      "seed":SEED,"training_region_columns":[0,383],"testing_region_columns":[384,511],
      "n_training_patches_per_texture":713,"n_heldout_patches_per_texture":217,
      "model":"DCGAN 32x32 grayscale; separate model per texture/condition",
      "treatment":"random horizontal flip of real training patches only with p=0.5",
      "control":"no augmentation","steps_per_model":args.steps,
      "limitations":["No industrial defect labels","One source photo per texture",
        "Overlapping patch correlation","Single seed, no confidence interval",
        "Descriptive feature distance, not FID"]}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(rows,indent=2))
if __name__=="__main__": main()
