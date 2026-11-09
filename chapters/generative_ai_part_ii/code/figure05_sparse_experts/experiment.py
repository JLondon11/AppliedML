"""Measured top-2 sparse MoE inference on the real sklearn handwritten-digit dataset."""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Linear(64,4)
        self.experts=nn.ModuleList([
            nn.Sequential(nn.Linear(64,64),nn.ReLU(),
                          nn.Linear(64,32),nn.ReLU()) for _ in range(4)])
        self.head=nn.Linear(32,10)

    def forward(self,x,details=False):
        scores=self.gate(x).softmax(-1)
        weights,ids=scores.topk(2,dim=1)
        sparse=torch.zeros_like(scores).scatter(1,ids,weights)
        sparse=sparse/sparse.sum(-1,keepdim=True)
        expert_vectors=torch.stack([expert(x) for expert in self.experts],dim=1)
        logits=self.head((expert_vectors*sparse[:,:,None]).sum(1))
        return (logits,scores,sparse,ids) if details else logits


def run(out,epochs=45,seed=31):
    if epochs<1:raise ValueError("epochs must be positive")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    data=load_digits()
    X=torch.from_numpy(data.data.astype("float32")/16)
    y=data.target.astype("int64")
    indices=np.arange(len(data.data))
    train_val,test=train_test_split(indices,test_size=270,random_state=seed,stratify=y)
    train,validation=train_test_split(train_val,test_size=270,random_state=seed,
                                      stratify=y[train_val])
    assert (len(train),len(validation),len(test))==(1257,270,270)
    assert not(set(train)&set(validation) or set(train)&set(test) or set(validation)&set(test))
    model=MoE();optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    gen=torch.Generator().manual_seed(seed+1)
    history=[];best=-1.;best_epoch=0
    for epoch in range(1,epochs+1):
        model.train();total=0
        for batch in torch.randperm(len(train),generator=gen).split(128):
            xb=X[train[batch]]
            yb=torch.as_tensor(y[train[batch]])
            optimizer.zero_grad()
            logits,probs,sparse,chosen=model(xb,details=True)
            loss=F.cross_entropy(logits,yb)+0.02*((probs.mean(0)-.25)**2).sum()
            loss.backward();optimizer.step();total+=loss.item()*len(batch)
        model.eval()
        with torch.no_grad():
            accuracy=float(accuracy_score(y[validation],model(X[validation]).argmax(1).numpy()))
        history.append({"epoch":epoch,"train_objective":total/len(train),
                        "validation_accuracy":accuracy})
        if accuracy>best:
            best=accuracy;best_epoch=epoch
            torch.save({"state":model.state_dict(),"seed":seed,
                        "epoch":epoch},out/"moe_best.pt")
    model.load_state_dict(torch.load(out/"moe_best.pt",map_location="cpu",
                                      weights_only=True)["state"])
    model.eval()
    with torch.no_grad():
        logits,probabilities,sparse,experts=model(X[test],details=True)
    predicted=logits.argmax(1).numpy()
    assert (sparse>0).sum(1).eq(2).all()
    assert torch.allclose(sparse.sum(1),torch.ones(len(test)),atol=1e-6)
    loads=(sparse>0).float().mean(0).numpy()
    avg_probs=probabilities.mean(0).numpy()
    correct=np.flatnonzero(predicted==y[test]);i=int(correct[0]) if len(correct) else 0
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.4))
    axes[0].imshow(X[test[i]].reshape(8,8),cmap="gray",vmin=0,vmax=1)
    axes[0].set_title("Real held-out digit")
    axes[0].axis("off")
    xx=np.arange(4)
    axes[1].bar(xx-.17,probabilities[i].numpy(),width=.34,label="Full gate softmax")
    axes[1].bar(xx+.17,sparse[i].numpy(),width=.34,label="Selected top-2 weights")
    axes[1].set(xticks=xx,xlabel="Expert index",ylabel="Routing weight",
                title="Actual sample-level routing")
    axes[1].legend(fontsize=7)
    axes[2].bar(xx-.17,avg_probs,width=.34,label="Mean gate probability")
    axes[2].bar(xx+.17,loads,width=.34,label="Expert selection fraction")
    axes[2].set(xticks=xx,xlabel="Expert index",ylabel="Probability / fraction",
                title="270 independent test images")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"Figure_005_Sparse_Expert_Routing.png",dpi=190,bbox_inches="tight")
    plt.close(fig)
    with (out/"test_predictions_routes.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["source_index","ground_truth","predicted","expert_1","expert_2",
                         "weight_1","weight_2"])
        writer.writerows([[int(test[j]),int(y[test[j]]),int(predicted[j]),
                           int(experts[j,0]),int(experts[j,1]),
                           float(sparse[j,experts[j,0]]),
                           float(sparse[j,experts[j,1]])]
                          for j in range(len(test))])
    m={
        "dataset_name":"scikit-learn real handwritten digits",
        "dataset_source_url":"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        "split_counts":{"train":1257,"validation":270,"test":270},
        "split_disjoint":True,"seed":seed,"epochs":epochs,"selected_epoch":best_epoch,
        "validation_accuracy":best,
        "independent_test_accuracy":float(accuracy_score(y[test],predicted)),
        "test_confusion_matrix":confusion_matrix(y[test],predicted,labels=range(10)).tolist(),
        "selected_experts_per_sample":2,"expert_count":4,
        "mean_routing_probabilities":avg_probs.tolist(),
        "expert_selection_fraction":loads.tolist(),
        "max_sparse_row_sum_error":float((sparse.sum(1)-1).abs().max()),
        "method":"top-2 sparse mixture-of-experts classifier on real handwritten-digit images",
        "limitations":"Not a foundation-model MoE; routing measured on 270 real held-out images. Expert selection fractions sum to 2."
    }
    (out/"metrics.json").write_text(json.dumps(m,indent=2)+"\n")
    print(json.dumps(m,indent=2))
    return m


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=45)
    a=p.parse_args()
    run(a.out,a.epochs)
