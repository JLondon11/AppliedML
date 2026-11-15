"""Reproduce Chapter 2 Figures 2.19 and 2.35 on authentic scikit-learn Digits data.

One observed Digits image undergoes three fixed, nonlearned 3x3 correlations.
Twelve independent 24-24 MLP fits use the same original-image train/test split.
No synthetic two-moons samples and no test-set-driven seed selection.
Run: OPENBLAS_NUM_THREADS=1 python run.py --out outputs
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
NAVY="#203E56";TEAL="#267568";COPPER="#BD7750";SLATE="#8295A2"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
def save(fig,out,stem):
    for extension in ("png","pdf","svg"):
        fig.savefig(out/(stem+"."+extension),dpi=230,bbox_inches="tight")
    plt.close(fig)
def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=load_digits();images=d.images.astype(np.float64);y=d.target
    digest=hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()
    idx=0;image=images[idx]
    kernels={"Horizontal derivative":np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=float)/8,
             "Vertical derivative":np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=float)/8,
             "Local average":np.ones((3,3))/9}
    maps={name:correlate(image,k,mode="reflect") for name,k in kernels.items()}
    for name,k in kernels.items():
        source_patch=np.pad(image,1,mode="symmetric")[3:6,4:7]
        assert np.isclose(maps[name][3,4],np.sum(source_patch*k))
    fig,ax=plt.subplots(2,4,figsize=(12.5,5.3),layout="constrained",
        gridspec_kw={"height_ratios":[1.35,.65]})
    panels=[("Original observed digit",image)]+list(maps.items())
    for i,(name,array) in enumerate(panels):
        limit=max(abs(array.min()),abs(array.max()))
        low,high=(-limit,limit) if i in (1,2) else (0,16)
        im=ax[0,i].imshow(array,cmap="RdBu_r" if i in (1,2) else "gray_r",
             vmin=low,vmax=high,interpolation="nearest")
        ax[0,i].set(title=name,xticks=range(8),yticks=range(8) if i==0 else [])
        ax[0,i].tick_params(labelsize=7)
        fig.colorbar(im,ax=ax[0,i],fraction=.047,pad=.02)
    ax[1,0].text(.05,.6,f"Original Digits record #{idx}\nGround-truth label: {y[idx]}\n"
                        "Image: 8×8 recorded pixels\nIntensity: 0–16",va="top",fontsize=9)
    ax[1,0].set_axis_off()
    for i,(name,k) in enumerate(kernels.items(),1):
        bound=max(abs(k).max(),.01)
        ax[1,i].imshow(k,cmap="RdBu_r",vmin=-bound,vmax=bound)
        ax[1,i].set(title=f"Fixed kernel · mean |response| {np.mean(abs(maps[name])):.3f}",
                    xticks=[],yticks=[])
        for (r,c),v in np.ndenumerate(k):
            ax[1,i].text(c,r,f"{v:.2f}",ha="center",va="center",fontsize=8,
                color=NAVY if abs(v)<.2 else "white")
    fig.suptitle("Real handwritten digit: exact fixed-kernel spatial responses",
        fontsize=15,fontweight="bold",color=NAVY)
    save(fig,out,"Figure_2_19_convolution_responses")
    tr,te=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=1729)
    scores=[];predictions={};histories={}
    for seed in range(12):
        model=make_pipeline(StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(24,24),max_iter=220,
                          early_stopping=False,random_state=seed))
        model.fit(d.data[tr]/16.,y[tr])
        p=model.predict(d.data[te]/16.)
        row={"seed":seed,"accuracy":float(accuracy_score(y[te],p)),
             "macro_f1":float(f1_score(y[te],p,average="macro")),
             "epochs_run":int(model[-1].n_iter_),"converged":bool(model[-1].n_iter_<220)}
        scores.append(row);predictions[str(seed)]=p.tolist()
        histories[str(seed)]=[float(v) for v in model[-1].loss_curve_]
    acc=np.array([r["accuracy"] for r in scores])
    f1=np.array([r["macro_f1"] for r in scores])
    fig,ax=plt.subplots(1,3,figsize=(12.2,4.05),layout="constrained")
    ax[0].plot(range(12),acc,"o-",color=TEAL,lw=1.5,ms=7)
    ax[0].axhline(acc.mean(),ls="--",color=COPPER,lw=1.1,label=f"Mean {acc.mean():.4f}")
    ax[0].set(title="(a) Twelve original-data evaluations",
        xlabel="MLP initialization seed",ylabel="Held-out accuracy",xticks=range(12))
    ax[0].legend(frameon=False,fontsize=8)
    low=int(acc.argmin());high=int(acc.argmax())
    for i in range(12):
        curve=histories[str(i)]
        color=SLATE if i not in (low,high) else (COPPER if i==low else TEAL)
        ax[1].plot(range(1,len(curve)+1),curve,color=color,
            alpha=.32 if i not in (low,high) else 1,
            lw=.9 if i not in (low,high) else 1.7)
    ax[1].set(title="(b) Actually recorded training histories",
        xlabel="Optimization iteration",ylabel="Cross-entropy training loss",yscale="log")
    ax[2].scatter(acc,f1,color=TEAL,alpha=.85,s=51)
    for i,(a,b) in enumerate(zip(acc,f1)):
        ax[2].annotate(str(i),(a,b),xytext=(4,4),textcoords="offset points",fontsize=7)
    ax[2].set(title="(c) Accuracy and macro-F1 per seed",
        xlabel="Held-out accuracy",ylabel="Held-out macro-F1")
    for a in ax:a.grid(alpha=.16)
    fig.suptitle("Training-seed variation on 1,797 authentic handwritten-digit images",
        fontsize=14.2,color=NAVY,fontweight="bold")
    save(fig,out,"Figure_2_35_seed_variation")
    evidence={"dataset":"Original scikit-learn Digits (not MNIST)",
      "dataset_url":"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
      "observations":len(y),"image_pixels_sha256_float64":digest,
      "image_index":idx,"image_label":int(y[idx]),"image_pixels":image.tolist(),
      "kernel_boundary":"scipy.ndimage.correlate reflect",
      "kernels":{k:v.tolist() for k,v in kernels.items()},
      "response_maps":{k:v.tolist() for k,v in maps.items()},
      "mean_absolute_responses":{k:float(abs(v).mean()) for k,v in maps.items()},
      "split_seed":1729,"training_images":len(tr),"test_images":len(te),
      "train_indices":tr.tolist(),"test_indices":te.tolist(),
      "model":"StandardScaler training-only, MLP 24-24, maximum 220 solver iterations",
      "seeds":list(range(12)),"test_ground_truth":y[te].tolist(),
      "run_metrics":scores,"run_predictions":predictions,"run_training_losses":histories,
      "accuracy_mean":float(acc.mean()),"accuracy_sd_sample":float(acc.std(ddof=1)),
      "accuracy_min":float(acc.min()),"accuracy_max":float(acc.max()),
      "macro_f1_mean":float(f1.mean())}
    assert len(tr)==1347 and len(te)==450
    (out/"real_digits_seed_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps({key:evidence[key] for key in
        ("observations","training_images","test_images","accuracy_mean",
         "accuracy_sd_sample","accuracy_min","accuracy_max")},indent=2))
    return evidence
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="outputs")
    run(p.parse_args().out)
