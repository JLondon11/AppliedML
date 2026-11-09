"""Chapter 9 Figure 4: measured causal KV caching on real digit-row sequences.

Cached and uncached inference share exactly the same trained model. CPU timing
is a local demonstration and is NOT a claim of production LLM speedup.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CachedRowPredictor(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.e=nn.Linear(8,width)
        self.p=nn.Embedding(8,width)
        self.q=nn.Linear(width,width,bias=False)
        self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False)
        self.out=nn.Linear(width,8)
        self.width=width

    def emb(self,x,offset=0):
        return self.e(x)+self.p(torch.arange(offset,offset+x.shape[1]))[None,:,:]

    def full(self,x):
        h=self.emb(x)
        q,k,v=self.q(h),self.k(h),self.v(h)
        score=q@k.transpose(-1,-2)/self.width**.5
        n=x.shape[1]
        score=score.masked_fill(torch.triu(torch.ones(n,n,dtype=torch.bool),diagonal=1),-torch.inf)
        return self.out(score.softmax(-1)@v)

    def cached(self,x):
        keys=[];values=[];outputs=[]
        for t in range(x.shape[1]):
            h=self.emb(x[:,t:t+1],offset=t)
            keys.append(self.k(h));values.append(self.v(h))
            k=torch.cat(keys,dim=1);v=torch.cat(values,dim=1);q=self.q(h)
            attention=(q@k.transpose(-1,-2)/self.width**.5).softmax(-1)
            outputs.append(self.out(attention@v))
        return torch.cat(outputs,dim=1)


def run(out,epochs=30,seed=23):
    if epochs<1:raise ValueError("epochs must be positive")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    dataset=load_digits()
    X=(dataset.data.astype("float32")/16).reshape(-1,8,8)
    y=dataset.target
    idx=np.arange(len(X))
    trv,te=train_test_split(idx,test_size=270,random_state=seed,stratify=y)
    tr,va=train_test_split(trv,test_size=270,random_state=seed,stratify=y[trv])
    assert (len(tr),len(va),len(te))==(1257,270,270)
    assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    x=torch.from_numpy(X)
    model=CachedRowPredictor()
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    generator=torch.Generator().manual_seed(seed+1)
    best=float("inf");history=[]
    for epoch in range(1,epochs+1):
        model.train();total=0
        for ids in torch.randperm(len(tr),generator=generator).split(128):
            xb=x[tr[ids]]
            optimizer.zero_grad()
            loss=F.mse_loss(model.full(xb[:,:-1]),xb[:,1:])
            loss.backward();optimizer.step()
            total+=loss.item()*len(ids)
        model.eval()
        with torch.no_grad():
            validation_mse=float(F.mse_loss(model.full(x[va,:-1]),x[va,1:]))
        history.append({"epoch":epoch,"train_next_row_mse":total/len(tr),
                        "validation_next_row_mse":validation_mse})
        if validation_mse<best:
            best=validation_mse;best_epoch=epoch
            torch.save({"state_dict":model.state_dict(),"epoch":epoch,"seed":seed},out/"cache_model.pt")
    model.load_state_dict(torch.load(out/"cache_model.pt",map_location="cpu",weights_only=True)["state_dict"])
    model.eval()
    with torch.no_grad():
        uncached=model.full(x[te,:-1])
        cached=model.cached(x[te,:-1])
        test_mse=float(F.mse_loss(uncached,x[te,1:]))
        max_error=float((uncached-cached).abs().max())
    assert max_error<1e-5

    testbatch=x[te[:64],:-1]
    def timing(fn):
        durations=[]
        with torch.no_grad():
            fn(testbatch)
            for _ in range(100):
                t0=time.perf_counter_ns()
                fn(testbatch)
                durations.append((time.perf_counter_ns()-t0)/1e6)
        return {"median_ms":float(np.median(durations)),
                "p10_ms":float(np.percentile(durations,10)),
                "p90_ms":float(np.percentile(durations,90))}
    t_full=timing(model.full)
    t_cached=timing(model.cached)
    lengths=np.arange(1,9)
    memory_bytes=2*lengths*model.width*4
    fig,axs=plt.subplots(1,3,figsize=(11.5,3.5))
    axs[0].imshow(X[te[0]],cmap="gray",vmin=0,vmax=1)
    axs[0].set_title("Real held-out digit")
    axs[0].axis("off")
    axs[1].plot(lengths,memory_bytes,"o-")
    axs[1].set(xlabel="Observed row tokens",ylabel="K+V cache bytes/sequence",
               title="Actual tensor dimensionality")
    axs[2].bar(["Full sequence\ncausal attention","Cached per-token\nattention"],
               [t_full["median_ms"],t_cached["median_ms"]])
    axs[2].set(ylabel="Median CPU time (ms)",title="64 real held-out sequences")
    fig.tight_layout()
    fig.savefig(out/"Figure_004_KV_Cache_Memory_Hierarchy.png",dpi=190,bbox_inches="tight")
    plt.close(fig)
    metrics={
        "dataset_name":"scikit-learn real handwritten digits",
        "dataset_source_url":"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        "split_counts":{"train":1257,"validation":270,"test":270},
        "split_disjoint":True,"seed":seed,"epochs":epochs,
        "best_epoch":best_epoch,"validation_next_row_mse":best,
        "independent_test_next_row_mse":test_mse,
        "cached_vs_full_max_abs_difference":max_error,
        "cache_bytes_per_sequence_8_tokens":int(memory_bytes[-1]),
        "batch_size_inference":64,"cpu_benchmark_repeats":100,
        "uncached_timing":t_full,"cached_timing":t_cached,
        "method":"single-head causal attention, measured KV caching on real digit row sequences",
        "limitation":"Tiny 7-token row sequence on CPU; cached Python loop may be slower. Not an LLM long-context or GPU benchmark."
    }
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))
    return metrics


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=30)
    a=parser.parse_args()
    run(a.out,a.epochs)
