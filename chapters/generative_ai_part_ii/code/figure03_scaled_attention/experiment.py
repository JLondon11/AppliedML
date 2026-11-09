"""Chapter 9 Figure 3: measured Q/K/V attention from real scikit-learn digits.

Rows of each real 8x8 image are the eight sequence tokens. This is a small
one-head image-row attention classifier, NOT a pretrained language model.
"""
from pathlib import Path
import argparse, csv, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


class RowAttentionClassifier(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.embedding=nn.Linear(8,width)
        self.q=nn.Linear(width,width,bias=False)
        self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False)
        self.projection=nn.Linear(width,width)
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,10))
        self.width=width

    def forward(self,x,explain=False):
        h=self.embedding(x.reshape(-1,8,8))
        q,k,v=self.q(h),self.k(h),self.v(h)
        scores=q@k.transpose(-2,-1)/self.width**.5
        weights=scores.softmax(-1)
        logits=self.output(self.projection(weights@v).mean(1))
        return (logits,weights,scores,q,k,v) if explain else logits


def run(out: Path, epochs=45, seed=19):
    if epochs<1:raise ValueError("epochs must be >=1")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    dataset=load_digits()
    X=(dataset.data.astype("float32")/16).reshape(-1,8,8)
    y=dataset.target.astype("int64")
    all_ids=np.arange(len(X))
    trv,te=train_test_split(all_ids,test_size=270,random_state=seed,stratify=y)
    tr,va=train_test_split(trv,test_size=270,random_state=seed,stratify=y[trv])
    assert (len(tr),len(va),len(te))==(1257,270,270)
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    xtr=torch.from_numpy(X[tr]);xva=torch.from_numpy(X[va]);xte=torch.from_numpy(X[te])
    model=RowAttentionClassifier()
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    generator=torch.Generator().manual_seed(seed+1)
    history=[];best=-1.;best_epoch=0
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(tr),generator=generator);total=0.
        for ids in order.split(128):
            optimizer.zero_grad()
            loss=F.cross_entropy(model(xtr[ids]),torch.tensor(y[tr][ids]))
            loss.backward();optimizer.step();total+=loss.item()*len(ids)
        model.eval()
        with torch.no_grad():
            acc=float(accuracy_score(y[va],model(xva).argmax(1).numpy()))
        history.append({"epoch":epoch,"train_cross_entropy":total/len(tr),
                        "validation_accuracy":acc})
        if acc>best:
            best=acc;best_epoch=epoch
            torch.save({"state_dict":model.state_dict(),"seed":seed,
                        "epoch":epoch},out/"attention_best.pt")
    checkpoint=torch.load(out/"attention_best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["state_dict"]);model.eval()
    with torch.no_grad():
        logits,w,score,q,k,v=model(xte,explain=True)
    pred=logits.argmax(1).numpy()
    assert w.shape==(270,8,8)
    assert torch.allclose(w.sum(-1),torch.ones(270,8),atol=1e-5)
    assert torch.allclose(score,q@k.transpose(-2,-1)/model.width**.5,atol=1e-5)
    correct=np.flatnonzero(pred==y[te])
    i=int(correct[0]) if len(correct) else 0
    fig,axs=plt.subplots(1,4,figsize=(12,3.3))
    axs[0].imshow(xte[i],cmap="gray",vmin=0,vmax=1)
    axs[0].set_title("Real held-out digit",fontsize=10)
    for ax,matrix,title in zip(
        axs[1:],[score[i].numpy(),w[i].numpy(),w.mean(0).numpy()],
        ["Measured scaled QKᵀ","Softmax attention","Mean attention, 270 test digits"]):
        ax.imshow(matrix,cmap="viridis",aspect="equal")
        ax.set(xlabel="Key image-row index",ylabel="Query image-row index")
        ax.set_title(title,fontsize=9)
    fig.tight_layout()
    fig.savefig(out/"Figure_003_Scaled_Dot_Product_Attention.png",dpi=220,bbox_inches="tight")
    fig.savefig(out/"Figure_003_Scaled_Dot_Product_Attention.svg",bbox_inches="tight")
    plt.close(fig)
    with (out/"training_history.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]))
        writer.writeheader();writer.writerows(history)
    with (out/"test_predictions.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["source_index","ground_truth","predicted"])
        writer.writerows(zip(te,y[te],pred))
    metrics={
       "dataset_name":"scikit-learn digits (real measured handwritten images)",
       "dataset_source_url":"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
       "split_counts":{"train":1257,"validation":270,"test":270},
       "split_disjoint":True,"seed":seed,"epochs":epochs,"selected_epoch":best_epoch,
       "validation_accuracy":best,"independent_test_accuracy":float(accuracy_score(y[te],pred)),
       "correct_test_count":int(np.sum(pred==y[te])),
       "confusion_matrix":confusion_matrix(y[te],pred,labels=np.arange(10)).tolist(),
       "mean_attention_matrix":w.mean(0).numpy().tolist(),
       "attention_row_sum_max_abs_error":float((w.sum(-1)-1).abs().max()),
       "scaled_qk_max_abs_error":float((score-q@k.transpose(-2,-1)/model.width**.5).abs().max()),
       "sample_source_index":int(te[i]),
       "method":"one-head scaled dot-product attention classifier on 8 image-row tokens",
       "limitation":"Small real-digit image-row attention, not a pretrained NLP Transformer."
    }
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    return metrics


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=45)
    args=p.parse_args()
    print(json.dumps(run(args.out,args.epochs),indent=2))
