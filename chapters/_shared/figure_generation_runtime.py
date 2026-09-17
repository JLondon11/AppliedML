"""Shared deterministic figure-generation runtime for AppliedML.

The chapter-local generators pass section-aware figure specs into this module. The
runtime never invents published benchmark values: evidence-gated figures raise unless
backed by a specialized verified-data generator stored in the chapter directory.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE=["#294C60","#5A7D7C","#B26E3B","#7B6D8D","#567D46","#9A6B5B"]
plt.rcParams.update({"font.size":8.5,"axes.labelsize":8.5,"legend.fontsize":7.5,
                     "xtick.labelsize":7.5,"ytick.labelsize":7.5,"font.family":"DejaVu Sans"})

def _seed(ch,n,key,declared):
    return int(declared) if int(declared) else int(hashlib.sha256(f"{ch}|{n}|{key}".encode()).hexdigest()[:8],16)

def _panel(ax,s): ax.text(.5,-.10,s,transform=ax.transAxes,ha="center",va="top")
def _save(fig,p): p.parent.mkdir(parents=True,exist_ok=True); fig.savefig(p,dpi=220,bbox_inches="tight",pad_inches=.03); plt.close(fig)

def _matrix(p,rng):
    A=rng.normal(size=(24,24))
    for _ in range(2): A=(A+np.roll(A,1,0)+np.roll(A,-1,0)+np.roll(A,1,1)+np.roll(A,-1,1))/5
    u,s,v=np.linalg.svd(A,full_matrices=False); R=np.outer(u[:,0],v[0])*s[0]
    fig,axs=plt.subplots(1,3,figsize=(8.7,3))
    for i,(ax,M) in enumerate(zip(axs,[A,R,A-R])): ax.imshow(M,cmap="coolwarm",aspect="auto"); ax.set_xlabel("feature index"); ax.set_ylabel("feature index"); _panel(ax,f"({chr(97+i)})")
    fig.tight_layout(); _save(fig,p)

def _surface(p,rng):
    x=np.linspace(-2.5,2.5,180); y=np.linspace(-2.2,2.2,160); X,Y=np.meshgrid(x,y); Z=.25*X**2+.55*Y**2+.18*np.sin(3*X)*np.cos(2*Y)
    fig,axs=plt.subplots(1,2,figsize=(7.7,3.1)); axs[0].contourf(X,Y,Z,20,cmap="viridis"); path=np.column_stack([np.linspace(2.1,0,18),np.linspace(-1.6,0,18)])+rng.normal(0,.07,(18,2)); axs[0].plot(path[:,0],path[:,1],"-o",ms=2.5); axs[0].set_xlabel("parameter 1");axs[0].set_ylabel("parameter 2");_panel(axs[0],"(a)")
    it=np.arange(1,61);axs[1].plot(it,np.exp(-it/15)+.03*np.sin(it/2));axs[1].set_xlabel("iteration");axs[1].set_ylabel("objective");_panel(axs[1],"(b)");fig.tight_layout();_save(fig,p)

def _distribution(p):
    x=np.linspace(-4,4,500);fig,ax=plt.subplots(figsize=(6.3,3.5))
    for m,s in [(-1.2,.7),(.4,.9),(1.6,.55)]:ax.plot(x,np.exp(-(x-m)**2/(2*s*s))/(s*np.sqrt(2*np.pi)))
    ax.set_xlabel("state / parameter");ax.set_ylabel("density");fig.tight_layout();_save(fig,p)

def _flow(p):
    x=np.linspace(-2,2,25);y=np.linspace(-1.5,1.5,20);X,Y=np.meshgrid(x,y);U=.18+.10*np.cos(2*Y);V=.12*np.sin(1.5*X)
    fig,ax=plt.subplots(figsize=(6.3,3.5));ax.streamplot(X,Y,U,V,density=1.1);ax.set_xlabel("x");ax.set_ylabel("y");ax.set_aspect("equal");fig.tight_layout();_save(fig,p)

def _architecture(p):
    d=np.arange(7);w=np.array([1,8,16,32,48,32,8]);r=np.array([1,3,5,9,17,25,33]);fig,axs=plt.subplots(1,2,figsize=(7.5,3))
    axs[0].plot(d,w,marker="o");axs[0].set_xlabel("stage");axs[0].set_ylabel("feature width");_panel(axs[0],"(a)")
    axs[1].plot(d,r,marker="s");axs[1].set_xlabel("stage");axs[1].set_ylabel("receptive field / context");_panel(axs[1],"(b)");fig.tight_layout();_save(fig,p)

def _rollout(p):
    n=12;goal=(10,10);haz={(5,5),(5,6),(6,5),(8,4)};V=np.zeros((n,n));g=.96
    for _ in range(220):
        N=V.copy()
        for i in range(n):
            for j in range(n):
                if (i,j)==goal:N[i,j]=1;continue
                vals=[]
                for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                    a=min(max(i+di,0),n-1);b=min(max(j+dj,0),n-1);r=-.02-(.6 if (a,b) in haz else 0)+(1 if (a,b)==goal else 0);vals.append(r+g*V[a,b])
                N[i,j]=max(vals)
        V=N
    fig,ax=plt.subplots(figsize=(5.2,4));ax.imshow(V,origin="lower",cmap="viridis");ax.set_xlabel("grid x");ax.set_ylabel("grid y");fig.tight_layout();_save(fig,p)

def _simulator(p,key):
    t=np.linspace(0,50,500);key=key.lower()
    if any(k in key for k in ("uav","drone","flight")):
        x=.8*t;y=8*np.sin(t/6);z=12+3*np.sin(t/4);fig=plt.figure(figsize=(6.4,3.8));ax=fig.add_subplot(111,projection="3d");ax.plot(x,y,z);ax.set_xlabel("x (m)");ax.set_ylabel("y (m)");ax.set_zlabel("z (m)");_save(fig,p);return
    y=np.sin(t/5)+.18*np.sin(t*1.7);est=np.convolve(y,np.ones(15)/15,mode="same");fig,ax=plt.subplots(figsize=(6.4,3.5));ax.plot(t,y,alpha=.55,label="observation");ax.plot(t,est,label="model state");ax.legend(frameon=False);ax.set_xlabel("time");fig.tight_layout();_save(fig,p)

def _public_data(p,key,rng):
    try:
        from sklearn.datasets import load_breast_cancer,load_digits
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix,roc_curve
    except Exception as e: raise RuntimeError("scikit-learn required") from e
    if any(k in key.lower() for k in ("digit","mnist","embedding","latent")):
        d=load_digits();X=d.data-d.data.mean(0);U,S,V=np.linalg.svd(X,full_matrices=False);Z=U[:,:2]*S[:2];fig,ax=plt.subplots(figsize=(6.1,4));ax.scatter(Z[:,0],Z[:,1],c=d.target,cmap="tab10",s=6,alpha=.7,linewidths=0);ax.set_xlabel("component 1");ax.set_ylabel("component 2");fig.tight_layout();_save(fig,p);return
    X,y=load_breast_cancer(return_X_y=True);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=int(rng.integers(1,100000)),stratify=y);m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=600));m.fit(Xtr,ytr)
    if "confusion" in key.lower():
        cm=confusion_matrix(yte,m.predict(Xte));fig,ax=plt.subplots(figsize=(5.2,4));ax.imshow(cm,cmap="viridis");ax.set_xlabel("predicted");ax.set_ylabel("true");fig.tight_layout();_save(fig,p);return
    prob=m.predict_proba(Xte)[:,1];fpr,tpr,_=roc_curve(yte,prob);fig,ax=plt.subplots(figsize=(5.8,3.7));ax.plot(fpr,tpr);ax.plot([0,1],[0,1],"--",lw=1);ax.set_xlabel("false-positive rate");ax.set_ylabel("true-positive rate");fig.tight_layout();_save(fig,p)

def generate(chapter,spec,out):
    n,section,asset,cls,status,declared,key,sha=spec;seed=_seed(chapter,n,key,declared);rng=np.random.default_rng(seed);lk=key.lower()
    if cls=="PUBLISHED/DATASET EVIDENCE": raise RuntimeError(f"figure {n}: use a specialized verified-data generator for published/dataset evidence")
    if cls=="COMPUTED POLICY / ROLLOUT":return _rollout(out)
    if cls=="SIMULATOR / NUMERICAL MODEL":return _simulator(out,key)
    if cls=="REAL DATA / REPRODUCIBLE EXPERIMENT":
        if any(k in lk for k in ("digit","mnist","embedding","latent","breast cancer","wdbc")): return _public_data(out,key,rng)
        raise RuntimeError(f"figure {n}: a figure-specific verified-data generator is required")
    if any(k in lk for k in ("matrix","tensor","attention","convolution","feature map","activation")):return _matrix(out,rng)
    if any(k in lk for k in ("flow","motion","velocity","tracking")):return _flow(out)
    if any(k in lk for k in ("architecture","network","pipeline","transformer","cnn","rnn","lstm","gru","encoder","decoder")):return _architecture(out)
    if any(k in lk for k in ("distribution","probability","bayes","uncertainty","calibration")):return _distribution(out)
    if any(k in lk for k in ("trajectory","policy","value function","q-function")):return _rollout(out)
    return _surface(out,rng)

def run_cli(chapter,specs,here):
    p=argparse.ArgumentParser();p.add_argument("--figure",type=int);p.add_argument("--section");p.add_argument("--output-root",type=Path,default=here/"generated");a=p.parse_args()
    sel=[s for s in specs if (a.figure is None or s[0]==a.figure) and (a.section is None or s[1]==a.section)]
    if not sel:raise SystemExit("no matching frozen figure spec")
    for s in sel:
        out=a.output_root/Path(s[2]).name;generate(chapter,s,out);print(f"generated {chapter} figure {s[0]:03d} | {s[1]} | {out}")
