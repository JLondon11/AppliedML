"""CNN on real scikit-learn Digits. Three disjoint, stratified splits. NOT MNIST."""
import json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 20261105
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)
OUT = Path(__file__).parent / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
digits = load_digits()
X = (digits.images/16.0).astype("float32")[:, None, :, :]
y = digits.target.astype("int64")
indices = np.arange(len(y))
train_idx, rest_idx = train_test_split(indices, test_size=0.4, random_state=SEED, stratify=y)
val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=SEED+1, stratify=y[rest_idx])
assert not (set(train_idx)&set(val_idx) or set(train_idx)&set(test_idx) or set(val_idx)&set(test_idx))
Xt,yt=torch.tensor(X[train_idx]),torch.tensor(y[train_idx])
Xv,yv=torch.tensor(X[val_idx]),torch.tensor(y[val_idx])
Xe,ye=torch.tensor(X[test_idx]),torch.tensor(y[test_idx])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(24*4*4,64),
            nn.ReLU(),nn.Linear(64,10))
    def forward(self,x):
        return self.classifier(self.features(x))

model=CNN(); opt=torch.optim.Adam(model.parameters(),lr=0.002); ce=nn.CrossEntropyLoss()
best=float("inf"); best_epoch=0; best_weights=None; stale=0; hist=[]
for epoch in range(1,61):
    model.train()
    perm=torch.randperm(len(train_idx),generator=torch.Generator().manual_seed(SEED+epoch))
    for batch in perm.split(128):
        opt.zero_grad(); loss=ce(model(Xt[batch]),yt[batch]);loss.backward();opt.step()
    model.eval()
    with torch.inference_mode():
        tl=ce(model(Xt),yt).item();vl=ce(model(Xv),yv).item()
        va=float(accuracy_score(y[val_idx],model(Xv).argmax(1).numpy()))
    hist.append(dict(epoch=epoch,training_loss=tl,validation_loss=vl,validation_accuracy=va))
    if vl < best - 1e-5:
        best=vl;best_epoch=epoch
        best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
        stale=0
    else: stale+=1
    if stale>=8: break
model.load_state_dict(best_weights);model.eval()
with torch.inference_mode():
    prediction=model(Xe).argmax(1).numpy()
    test_loss=ce(model(Xe),ye).item()
    features=model.features[0](Xe[:1]).squeeze(0).numpy()
cm=confusion_matrix(y[test_idx],prediction,labels=np.arange(10))
metrics=dict(dataset="scikit-learn Digits 8x8; NOT MNIST",
    total_samples=len(y),training_samples=len(train_idx),validation_samples=len(val_idx),
    testing_samples=len(test_idx),seed=SEED,epochs_trained=len(hist),
    best_validation_epoch=best_epoch,test_accuracy=float(accuracy_score(y[test_idx],prediction)),
    test_macro_f1=float(f1_score(y[test_idx],prediction,average="macro")),
    correct_test_predictions=int((prediction==y[test_idx]).sum()),
    test_cross_entropy=float(test_loss),model_parameters=sum(p.numel() for p in model.parameters()))
(OUT/"results.json").write_text(json.dumps(dict(metrics=metrics,history=hist,confusion_matrix=cm.tolist(),
    train_indices=train_idx.tolist(),validation_indices=val_idx.tolist(),
    test_indices=test_idx.tolist()),indent=2))
torch.save(model.state_dict(),OUT/"model_state.pt")
plt.rcParams.update({"font.size":10,"axes.spines.right":False,"axes.spines.top":False})
fig,ax=plt.subplots(figsize=(6.5,3.7))
ax.plot([r["epoch"] for r in hist],[r["training_loss"] for r in hist],label="Training loss",lw=2)
ax.plot([r["epoch"] for r in hist],[r["validation_loss"] for r in hist],label="Validation loss",lw=2)
ax.axvline(best_epoch,ls=":",color="gray",label=f"Selected epoch {best_epoch}")
ax.set(xlabel="Epoch",ylabel="Cross-entropy loss");ax.legend(frameon=False);ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(OUT/"train_validation_loss.png",dpi=220);fig.savefig(OUT/"train_validation_loss.pdf");plt.close(fig)
fig,ax=plt.subplots(figsize=(5.3,4.6));im=ax.imshow(cm,cmap="Blues")
ax.set(xlabel="Predicted class",ylabel="True class",xticks=range(10),yticks=range(10))
fig.colorbar(im,ax=ax,label="Test-set count");fig.tight_layout()
fig.savefig(OUT/"test_confusion_matrix.png",dpi=220);fig.savefig(OUT/"test_confusion_matrix.pdf");plt.close(fig)
fig,axes=plt.subplots(3,6,figsize=(9.6,5.2))
for a,i in zip(axes.flat,range(18)):
    a.imshow(X[test_idx[i],0],cmap="gray_r",vmin=0,vmax=1)
    a.set_title(f"True {y[test_idx[i]]} · Pred {prediction[i]}",fontsize=8);a.axis("off")
fig.tight_layout();fig.savefig(OUT/"heldout_predictions.png",dpi=220);fig.savefig(OUT/"heldout_predictions.pdf");plt.close(fig)
fig,axes=plt.subplots(4,4,figsize=(6.4,6))
for i,a in enumerate(axes.flat):
    a.imshow(features[i],cmap="viridis");a.axis("off");a.set_title(f"Filter {i+1}",fontsize=8)
fig.tight_layout();fig.savefig(OUT/"feature_maps.png",dpi=220);fig.savefig(OUT/"feature_maps.pdf");plt.close(fig)
print(json.dumps(metrics,indent=2))
