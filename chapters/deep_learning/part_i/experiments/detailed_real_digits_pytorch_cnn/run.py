"""Measured original 8x8 Digits PyTorch CNN experiment for detailed Chapter 2.

Reproduces the 1,797 authentic observed handwritten images, 1077/270/450
train/validation/test partition, test CNN and matched logistic baseline.
Saves complete per-image predictions, probabilities, metrics and the
four-panel publication PNG/PDF/SVG. Not an MNIST or CIFAR-10 experiment.
Requires torch, numpy, scipy, sklearn, matplotlib, and their dependencies.
"""
from pathlib import Path
import argparse, copy, hashlib, json, random, platform
import numpy as np
import torch
from torch import nn
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 1729

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x):
        return self.classifier(self.features(x))

def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    data = load_digits()
    X = data.images.astype(np.float32) / 16.0
    y = data.target.astype(np.int64)
    ids = np.arange(len(y))
    trainval, test = train_test_split(
        ids, test_size=.25, stratify=y, random_state=SEED)
    train, val = train_test_split(
        trainval, test_size=.2, stratify=y[trainval], random_state=SEED)
    assert (len(train), len(val), len(test)) == (1077, 270, 450)
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    criterion = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X[:, None, :, :])
    yt = torch.tensor(y)
    history = []
    best_val = float("inf")
    best_epoch, best_model = None, None
    for epoch in range(1, 36):
        model.train()
        generator = torch.Generator().manual_seed(SEED * 100 + epoch)
        order = torch.tensor(train)[torch.randperm(len(train), generator=generator)]
        losses = []
        for b in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xt[b])
            loss = criterion(logits, yt[b])
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach()), len(b)))
        model.eval()
        with torch.no_grad():
            vlogits = model(Xt[val])
            v_loss = float(criterion(vlogits, yt[val]))
        mean_train = sum(value * n for value, n in losses) / len(train)
        history.append({"epoch":epoch,
            "train_cross_entropy":mean_train,
            "validation_cross_entropy":v_loss})
        if v_loss < best_val - 1e-6:
            best_val, best_epoch = v_loss, epoch
            best_model = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        logits = model(Xt[test])
        probs = torch.softmax(logits, dim=1).numpy()
    cnn_pred = probs.argmax(axis=1)
    scale = StandardScaler().fit(X[train].reshape(len(train), -1))
    baseline = LogisticRegression(max_iter=1500, random_state=SEED)
    baseline.fit(scale.transform(X[train].reshape(len(train), -1)), y[train])
    log_pred = baseline.predict(scale.transform(X[test].reshape(len(test), -1)))
    cm = confusion_matrix(y[test], cnn_pred, labels=np.arange(10))
    acc = float(accuracy_score(y[test], cnn_pred))
    baseacc = float(accuracy_score(y[test], log_pred))
    correct = cnn_pred == y[test]
    evidence = {
      "dataset":"Original 1797 observed 8x8 sklearn Digits grayscale images; NOT MNIST",
      "source_url":"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
      "source_float32_sha256":hashlib.sha256(X.tobytes()).hexdigest(),
      "image_count":len(y),"training_size":len(train),
      "validation_size":len(val),"test_size":len(test),
      "train_indices":train.tolist(),"validation_indices":val.tolist(),
      "test_indices":test.tolist(),"seed":SEED,
      "cnn_parameters":sum(p.numel() for p in model.parameters()),
      "train_method":"35 Adam epochs, select minimum validation cross-entropy, full test evaluated after selection",
      "best_validation_epoch":best_epoch,"training_history":history,
      "cnn_accuracy":acc,
      "cnn_macro_f1":float(f1_score(y[test],cnn_pred,average="macro")),
      "cnn_test_cross_entropy":float(log_loss(y[test],probs,labels=np.arange(10))),
      "logistic_baseline_test_accuracy":baseacc,
      "logistic_baseline_macro_f1":float(f1_score(y[test],log_pred,average="macro")),
      "test_labels":y[test].tolist(),
      "test_CNN_predict":cnn_pred.tolist(),"test_CNN_probabilities":probs.tolist(),
      "test_logistic_predict":log_pred.tolist(),"confusion_matrix":cm.tolist(),
      "number_of_test_errors":int((~correct).sum()),
      "limitations":"One original source, one held-out split, one seed; 8x8 Digits is not MNIST; no superiority claim"}
    (out/"original_digits_pytorch_cnn_test_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.3,
       "axes.spines.top":False,"axes.spines.right":False})
    fig=plt.figure(figsize=(12.3,8.6),layout="constrained")
    grid=fig.add_gridspec(2,2)
    ax0=fig.add_subplot(grid[0,0])
    ax1=fig.add_subplot(grid[0,1])
    ax2=fig.add_subplot(grid[1,0])
    ax3=fig.add_subplot(grid[1,1])
    samples=[int(test[np.where(y[test]==cl)[0][0]]) for cl in range(10)]
    tile=np.zeros((2*8,5*8))
    for k,idx in enumerate(samples):
        tile[(k//5)*8:(k//5+1)*8,(k%5)*8:(k%5+1)*8]=data.images[idx]
    ax0.imshow(tile,cmap="gray_r",vmin=0,vmax=16,interpolation="nearest")
    ax0.set(title="(a) Original test photographs (digits 0–9)",xticks=[],yticks=[])
    ax1.plot([z["epoch"] for z in history],
      [z["train_cross_entropy"] for z in history],
      c="#39546c",lw=1.6,label="Train")
    ax1.plot([z["epoch"] for z in history],
      [z["validation_cross_entropy"] for z in history],
      c="#B87751",lw=1.8,label="Validation")
    ax1.axvline(best_epoch,c="#247E76",lw=1.1,ls="--",label=f"Selected epoch {best_epoch}")
    ax1.set(title="(b) Executed optimization and validation",xlabel="Epoch",ylabel="Cross-entropy")
    ax1.legend(frameon=False,fontsize=8);ax1.grid(alpha=.16)
    im=ax2.imshow(cm,cmap="Blues",interpolation="nearest")
    ax2.set(title="(c) Actual original held-out predictions",xlabel="Predicted digit",
            ylabel="Observed digit",xticks=range(10),yticks=range(10))
    fig.colorbar(im,ax=ax2,fraction=.046,pad=.03,label="Test-image count")
    for k in range(10):
        ax2.text(k,k,str(cm[k,k]),ha="center",va="center",fontsize=8,
            color="white" if cm[k,k]>20 else "#15324d")
    x=np.arange(10)
    recall=np.diag(cm)/np.maximum(cm.sum(axis=1),1)
    ax3.plot(x,recall,color="#247E76",marker="o",lw=2,ms=6)
    ax3.axhline(acc,color="#B87751",lw=1.2,ls="--",label=f"CNN overall {acc:.3f}")
    ax3.axhline(baseacc,color="#39546c",lw=1.2,ls=":",label=f"Logistic baseline {baseacc:.3f}")
    ax3.set(title="(d) Class recall and fair held-out reference",
      xlabel="Digit class",ylabel="Recall / aggregate accuracy",xticks=x,ylim=(.5,1.035))
    ax3.legend(frameon=False,fontsize=8);ax3.grid(alpha=.16)
    fig.suptitle("PyTorch CNN: genuine handwritten-digit image classification",
       fontsize=15.5,fontweight="bold",color="#203D57")
    for ext in ("png","pdf","svg"):
        fig.savefig(out/("Figure_2_Observed_Digits_PyTorch_CNN."+ext),
            dpi=240,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"training":len(train),"validation":len(val),"testing":len(test),
        "cnn_accuracy":acc,"baseline_accuracy":baseacc,"best_epoch":best_epoch},indent=2))
    return evidence

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="outputs")
    run(p.parse_args().out)
