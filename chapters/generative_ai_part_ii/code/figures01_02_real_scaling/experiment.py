"""Chapter 9 Figures 1/2: measured small-MLP scaling on real digits.

15 real training runs, 5 model widths × 3 seeds. The FLOP quantity is an
architecture-derived proxy; it is NOT measured accelerator FLOPs or an LLM fit.
"""
import argparse,csv,json,time
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


def run(out, epochs=35, seeds=(11,17,23)):
    if epochs<1 or len(seeds)<2:
        raise ValueError("Expected epochs>0 and at least two seeds")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    dataset=load_digits()
    X=torch.tensor(dataset.data.astype("float32")/16)
    y=torch.tensor(dataset.target,dtype=torch.long)
    indices=np.arange(len(X))
    trv,te=train_test_split(indices,test_size=270,random_state=47,
                           stratify=y.numpy())
    tr,va=train_test_split(trv,test_size=270,random_state=47,
                           stratify=y[trv].numpy())
    assert (len(tr),len(va),len(te))==(1257,270,270)
    assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    results=[]
    for width in (8,16,32,64,128):
        for seed in seeds:
            torch.manual_seed(seed)
            model=nn.Sequential(nn.Linear(64,width),nn.ReLU(),
                                nn.Linear(width,width),nn.ReLU(),
                                nn.Linear(width,10))
            params=sum(p.numel() for p in model.parameters())
            optimizer=torch.optim.Adam(model.parameters(),lr=.002)
            generator=torch.Generator().manual_seed(seed+100)
            best=float("inf");best_weights=None;best_epoch=0
            start=time.perf_counter()
            for epoch in range(1,epochs+1):
                model.train()
                for batch in torch.randperm(len(tr),generator=generator).split(128):
                    optimizer.zero_grad()
                    loss=F.cross_entropy(model(X[tr[batch]]),y[tr[batch]])
                    loss.backward();optimizer.step()
                model.eval()
                with torch.no_grad():
                    val=float(F.cross_entropy(model(X[va]),y[va]))
                if val<best:
                    best=val;best_epoch=epoch
                    best_weights={k:v.clone() for k,v in model.state_dict().items()}
            elapsed=time.perf_counter()-start
            model.load_state_dict(best_weights);model.eval()
            with torch.no_grad():
                test=float(F.cross_entropy(model(X[te]),y[te]))
                acc=float((model(X[te]).argmax(1)==y[te]).float().mean())
            # 2 FLOPs/MAC forward; approximate forward+backward as 3× forward.
            flop_proxy=6*(64*width+width*width+width*10)*len(tr)*epochs
            results.append({
                "width":width,"seed":seed,"parameters":params,
                "best_epoch":best_epoch,"validation_ce":best,
                "independent_test_ce":test,"independent_test_accuracy":acc,
                "elapsed_train_seconds":elapsed,
                "nominal_train_flops_proxy":flop_proxy})
    with (out/"measured_runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    groups=[]
    for width in (8,16,32,64,128):
        subset=[x for x in results if x["width"]==width]
        groups.append({
            "width":width,"parameters":subset[0]["parameters"],
            "mean_test_ce":float(np.mean([r["independent_test_ce"] for r in subset])),
            "sd_test_ce":float(np.std([r["independent_test_ce"] for r in subset],ddof=1)),
            "mean_train_seconds":float(np.mean([r["elapsed_train_seconds"] for r in subset])),
            "mean_train_flops_proxy":float(np.mean([r["nominal_train_flops_proxy"] for r in subset])),
            "mean_test_accuracy":float(np.mean([r["independent_test_accuracy"] for r in subset]))})
    params=np.array([r["parameters"] for r in groups])
    losses=np.array([r["mean_test_ce"] for r in groups])
    errors=np.array([r["sd_test_ce"] for r in groups])
    flop=np.array([r["mean_train_flops_proxy"] for r in groups])
    seconds=np.array([r["mean_train_seconds"] for r in groups])
    fig,ax=plt.subplots(figsize=(7,3.9))
    ax.errorbar(params,losses,yerr=errors,fmt="o-",capsize=3)
    ax.set_xscale("log")
    ax.set(xlabel="Trained model parameters (log scale)",
           ylabel="Independent-test cross-entropy",
           title="Measured MLP capacity scaling on real handwritten digits")
    fig.tight_layout()
    fig.savefig(out/"Figure_001_Empirical_Scaling_Regimes.png",dpi=200)
    plt.close(fig)
    pareto=[not any(
        flop[j]<=flop[i] and losses[j]<=losses[i] and
        (flop[j]<flop[i] or losses[j]<losses[i])
        for j in range(len(groups)))
        for i in range(len(groups))]
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.7))
    axs[0].errorbar(flop,losses,yerr=errors,fmt="o",capsize=3,label="Actually trained MLPs")
    axs[0].plot(flop[pareto],losses[pareto],label="Measured nondominated set")
    axs[0].set_xscale("log")
    axs[0].set(xlabel="Training FLOPs proxy (derived)",
               ylabel="Independent-test cross-entropy",
               title="Real-data compute–quality comparison")
    axs[0].legend(fontsize=7)
    axs[1].scatter(seconds,losses)
    axs[1].set(xlabel="Measured training wall-clock seconds",
               ylabel="Independent-test cross-entropy",
               title="Observed CPU time versus quality")
    fig.tight_layout()
    fig.savefig(out/"Figure_002_Compute_Quality_Pareto_Frontier.png",dpi=200)
    plt.close(fig)
    metrics={
        "dataset_name":"scikit-learn real handwritten digits",
        "dataset_source_url":"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        "split_counts":{"train":1257,"validation":270,"test":270},
        "split_disjoint":True,"split_seed":47,
        "model_widths":[8,16,32,64,128],
        "training_seeds":list(seeds),"epochs":epochs,
        "trained_runs":len(results),"by_width":groups,
        "pareto_widths":[groups[i]["width"] for i,z in enumerate(pareto) if z],
        "method":"five real-data trained MLP widths, three seeds, independent test loss",
        "limitations":"Small handwritten-digit classification study, NOT Chinchilla or LLM scaling. Training FLOPs is an architecture-derived proxy; wall-clock measured on local CPU."}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))
    return metrics


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=35)
    a=p.parse_args()
    run(a.out,a.epochs)
