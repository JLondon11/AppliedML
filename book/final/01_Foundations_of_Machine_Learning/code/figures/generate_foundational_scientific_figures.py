from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, make_moons, make_blobs
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from skimage import data, color, filters
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

OUT = None
PALETTE = ["#355C7D","#4F7C6E","#B26E3B","#7A6AA6","#657A8A","#8B5E6B","#6E7D58","#4D6C73"]
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

def panel(ax, s):
    ax.text(.5, -.20, s, transform=ax.transAxes, ha="center", va="top", fontsize=10)

def save(fig, stem):
    fig.tight_layout()
    for ext in ("png","svg","pdf"):
        fig.savefig(OUT/f"{stem}.{ext}", dpi=300 if ext=="png" else None,
                    bbox_inches="tight", pad_inches=.03)
    plt.close(fig)

def fig_0103_model_capacity():
    x=np.linspace(-1,1,24)
    y=np.sin(2.5*np.pi*x)+0.18*RNG.normal(size=len(x))
    grid=np.linspace(-1.08,1.08,500)
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.2))
    for ax,deg,c in zip(axs,[1,5,18],[PALETTE[0],PALETTE[1],PALETTE[2]]):
        model=make_pipeline(PolynomialFeatures(deg),Ridge(alpha=1e-5))
        model.fit(x[:,None],y)
        ax.scatter(x,y,s=18,facecolor="white",edgecolor="#444444",linewidth=.7)
        ax.plot(grid,model.predict(grid[:,None]),lw=2,color=c)
        ax.set_ylim(-2,2); ax.set_xlabel("Input x"); ax.set_ylabel("Response")
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_03_model_capacity")

def fig_0104_inductive_bias():
    x=np.array([-1.0,-.55,-.1,.35,.8]); y=np.array([-.2,.7,.1,.9,.15])
    grid=np.linspace(-1.1,1.05,500)
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.25))
    # polynomial interpolant
    coef=np.polyfit(x,y,deg=len(x)-1)
    axs[0].scatter(x,y,s=25,facecolor="white",edgecolor="#333333")
    axs[0].plot(grid,np.polyval(coef,grid),lw=2,color=PALETTE[0])
    axs[0].set_xlabel("Input x"); axs[0].set_ylabel("f(x)")
    # RBF-like interpolation with tunable width
    for sigma,c in zip([.18,.35,.7],PALETTE[1:4]):
        K=np.exp(-((x[:,None]-x[None,:])**2)/(2*sigma**2))
        alpha=np.linalg.solve(K+1e-8*np.eye(len(x)),y)
        Kg=np.exp(-((grid[:,None]-x[None,:])**2)/(2*sigma**2))
        axs[1].plot(grid,Kg@alpha,lw=1.8,color=c,label=f"width={sigma}")
    axs[1].scatter(x,y,s=25,facecolor="white",edgecolor="#333333",zorder=5)
    axs[1].legend(frameon=False,fontsize=8); axs[1].set_xlabel("Input x"); axs[1].set_ylabel("f(x)")
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_04_inductive_bias")

def fig_0106_learning_curves():
    X,y=make_moons(n_samples=900,noise=.28,random_state=42)
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    train_sizes=np.linspace(.12,1.0,8)
    models=[
        ("underfit",make_pipeline(StandardScaler(),SVC(kernel="linear",C=.08))),
        ("well fit",make_pipeline(StandardScaler(),SVC(kernel="rbf",C=3.0,gamma=1.5))),
        ("high variance",DecisionTreeClassifier(random_state=42))
    ]
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.2))
    for ax,(name,model),accent in zip(axs,models,[PALETTE[0],PALETTE[1],PALETTE[2]]):
        n,tr,va=learning_curve(
            model,X,y,cv=cv,train_sizes=train_sizes,scoring="balanced_accuracy",
            shuffle=True,random_state=42,n_jobs=1
        )
        trm,trs=tr.mean(1),tr.std(1)
        vam,vas=va.mean(1),va.std(1)
        ax.plot(n,trm,marker="o",ms=3.5,lw=1.8,color=accent,label="training")
        ax.fill_between(n,trm-trs,trm+trs,color=accent,alpha=.12)
        ax.plot(n,vam,marker="o",ms=3.5,lw=1.8,color="#555555",label="cross-validation")
        ax.fill_between(n,vam-vas,vam+vas,color="#777777",alpha=.10)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(.5,1.02)
        ax.legend(frameon=False,fontsize=8)
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_06_learning_curves")

def fig_0107_bias_variance():
    grid=np.linspace(-1,1,150)
    ftrue=np.sin(np.pi*grid)
    degrees=[1,3,5,9,15]
    nrep=120
    noise_sigma=.25
    bias2=[]; var=[]
    for d in degrees:
        preds=[]
        for _ in range(nrep):
            x=RNG.uniform(-1,1,24)
            y=np.sin(np.pi*x)+RNG.normal(0,noise_sigma,len(x))
            model=make_pipeline(PolynomialFeatures(d),Ridge(alpha=1e-5))
            model.fit(x[:,None],y)
            preds.append(model.predict(grid[:,None]))
        P=np.vstack(preds)
        mean=P.mean(0)
        bias2.append(float(np.mean((mean-ftrue)**2)))
        var.append(float(np.mean(P.var(0))))
    noise=np.full(len(degrees),noise_sigma**2)
    total=np.asarray(bias2)+np.asarray(var)+noise
    fig,ax=plt.subplots(figsize=(7.2,3.6))
    ax.plot(degrees,bias2,marker="o",lw=1.9,label="squared bias",color=PALETTE[0])
    ax.plot(degrees,var,marker="o",lw=1.9,label="variance",color=PALETTE[2])
    ax.plot(degrees,noise,ls="--",lw=1.5,label="irreducible noise",color="#777777")
    ax.plot(degrees,total,marker="o",lw=2.2,label="expected test MSE",color=PALETTE[1])
    ax.set_xlabel("Polynomial degree (model capacity)")
    ax.set_ylabel("Bias--variance error component")
    ax.legend(frameon=False,ncol=2,fontsize=8)
    save(fig,"figure_01_07_bias_variance")

def fig_0108_representation_preprocessing():
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,stratify=y,random_state=42)
    scaler=StandardScaler().fit(Xtr)
    Xs=scaler.transform(Xtr)
    pca=PCA().fit(Xs)
    raw_scale=np.std(Xtr,axis=0)
    std_scale=np.std(Xs,axis=0)
    raw_gram=(Xtr.T@Xtr)/len(Xtr)+1e-8*np.eye(Xtr.shape[1])
    std_gram=(Xs.T@Xs)/len(Xs)+1e-8*np.eye(Xs.shape[1])
    cond_raw=float(np.linalg.cond(raw_gram)); cond_std=float(np.linalg.cond(std_gram))
    Xtr_p=pca.transform(Xs)[:,:10]
    Xte_s=scaler.transform(Xte); Xte_p=pca.transform(Xte_s)[:,:10]
    models=[
      ("raw",make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500))),
      ("standardized",LogisticRegression(max_iter=1500)),
      ("PCA-10",LogisticRegression(max_iter=1500))
    ]
    aucs=[]
    for name,m in models:
        if name=="raw": a,b=Xtr,Xte
        elif name=="standardized": a,b=Xs,Xte_s
        else: a,b=Xtr_p,Xte_p
        m.fit(a,ytr); aucs.append(roc_auc_score(yte,m.predict_proba(b)[:,1]))
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.15))
    order=np.argsort(raw_scale)
    axs[0].semilogy(np.arange(len(order)),raw_scale[order],marker="o",ms=3,lw=1.2,label="raw",color=PALETTE[2])
    axs[0].semilogy(np.arange(len(order)),std_scale[order],marker="o",ms=3,lw=1.2,label="standardized",color=PALETTE[0])
    axs[0].set_xlabel("Feature index (sorted by raw scale)"); axs[0].set_ylabel("Training-set standard deviation"); axs[0].legend(frameon=False,fontsize=8)
    axs[1].bar(["raw","standardized"],[cond_raw,cond_std],color=[PALETTE[2],PALETTE[1]])
    axs[1].set_yscale("log"); axs[1].set_ylabel("Condition number of regularized Gram matrix")
    cum=np.cumsum(pca.explained_variance_ratio_)
    axs[2].plot(np.arange(1,len(cum)+1),cum,lw=1.8,color=PALETTE[3],label="cumulative variance")
    ax2=axs[2].twinx()
    ax2.scatter([1,2,3],aucs,s=35,color=[PALETTE[0],PALETTE[1],PALETTE[2]],zorder=4)
    axs[2].set_xlabel("Principal components retained"); axs[2].set_ylabel("Cumulative explained variance")
    ax2.set_ylabel("Held-out ROC-AUC"); ax2.set_ylim(.90,1.0)
    axs[2].axvline(10,ls="--",lw=1,color="#777777")
    axs[2].text(10.5,.55,"PCA-10",fontsize=8)
    for i,a in enumerate(axs):
        a.spines[["top","right"]].set_visible(False); a.tick_params(direction="out")
        a.text(.5,-.23,f"({chr(97+i)})",transform=a.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=2.0)
    save(fig,"figure_01_08_representation_preprocessing")

def fig_0110_dimensionality():
    D=load_digits(); X=D.data[:700]; y=D.target[:700]
    embeds=[
        PCA(2,random_state=42).fit_transform(X),
        TSNE(2,random_state=42,init="pca",learning_rate="auto",perplexity=30,max_iter=700).fit_transform(X),
        Isomap(n_components=2,n_neighbors=12).fit_transform(X)
    ]
    fig,axs=plt.subplots(1,3,figsize=(9.6,3.0))
    for ax,E in zip(axs,embeds):
        ax.scatter(E[:,0],E[:,1],c=y,cmap="tab10",s=7,alpha=.75,linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_10_dimensionality")

def fig_0111_hpo():
    X,y=load_breast_cancer(return_X_y=True)
    cv=StratifiedKFold(4,shuffle=True,random_state=42)
    Cs=np.logspace(-4,3,36)
    scores=[]
    for C in Cs:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1500))
        s=cross_val_score(model,X,y,cv=cv,scoring="roc_auc").mean()
        scores.append(s)
    scores=np.array(scores)
    # matched 12-evaluation budgets
    random_idx=RNG.choice(len(Cs),12,replace=False)
    grid_idx=np.linspace(0,len(Cs)-1,12).round().astype(int)
    # coarse-to-fine: 6 coarse, then 6 around best coarse
    coarse=np.linspace(0,len(Cs)-1,6).round().astype(int)
    bestc=coarse[np.argmax(scores[coarse])]
    fine=np.unique(np.clip(np.arange(bestc-3,bestc+4),0,len(Cs)-1))[:6]
    half=np.r_[coarse,fine][:12]
    fig,axs=plt.subplots(1,2,figsize=(9.0,3.4))
    axs[0].plot(np.log10(Cs),scores,lw=1.8,color=PALETTE[0])
    axs[0].scatter(np.log10(Cs[random_idx]),scores[random_idx],s=25,label="random",color=PALETTE[2])
    axs[0].scatter(np.log10(Cs[grid_idx]),scores[grid_idx],s=22,label="grid",marker="s",color=PALETTE[1])
    axs[0].set_xlabel("log10(C)"); axs[0].set_ylabel("4-fold ROC-AUC"); axs[0].legend(frameon=False,fontsize=8)
    for idx,label,c in [(random_idx,"random",PALETTE[2]),(grid_idx,"grid",PALETTE[1]),(half,"coarse-to-fine",PALETTE[3])]:
        b=[]; cur=-np.inf
        for j in idx:
            cur=max(cur,scores[j]); b.append(cur)
        axs[1].plot(range(1,len(b)+1),b,marker="o",ms=3,lw=1.6,label=label,color=c)
    axs[1].set_xlabel("Evaluation count"); axs[1].set_ylabel("Best-so-far ROC-AUC"); axs[1].legend(frameon=False,fontsize=8)
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_11_hpo")

def fig_0112_metrics():
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,stratify=y,random_state=42)
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500)).fit(Xtr,ytr)
    p=clf.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
    fpr,tpr,_=roc_curve(yte,p); pr,rc,_=precision_recall_curve(yte,p)
    frac,mean=calibration_curve(yte,p,n_bins=8,strategy="quantile")
    cm=confusion_matrix(yte,pred)
    fig,axs=plt.subplots(1,4,figsize=(12.0,2.9))
    axs[0].plot(fpr,tpr,lw=2,color=PALETTE[0]); axs[0].plot([0,1],[0,1],"--",lw=1,color="#888")
    axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate")
    axs[1].plot(rc,pr,lw=2,color=PALETTE[1]); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    axs[2].plot(mean,frac,marker="o",lw=1.8,color=PALETTE[2]); axs[2].plot([0,1],[0,1],"--",lw=1,color="#888")
    axs[2].set_xlabel("Mean predicted probability"); axs[2].set_ylabel("Observed fraction")
    im=axs[3].imshow(cm,cmap="Blues",vmin=0,vmax=cm.max())
    for (i,j),v in np.ndenumerate(cm): axs[3].text(j,i,str(v),ha="center",va="center")
    axs[3].set_xticks([0,1],["0","1"]); axs[3].set_yticks([0,1],["0","1"])
    axs[3].set_xlabel("Predicted class"); axs[3].set_ylabel("True class")
    fig.colorbar(im,ax=axs[3],fraction=.046,pad=.04,label="Count")
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_12_metrics")

def fig_0113_cv():
    n=48
    y=np.array([0]*28+[1]*20)
    idx=np.arange(n)

    kf=KFold(6,shuffle=False)
    skf=StratifiedKFold(6,shuffle=True,random_state=42)
    M1=np.zeros((6,n),dtype=int)
    M2=np.zeros((6,n),dtype=int)
    for r,(_,te) in enumerate(kf.split(idx)):
        M1[r,te]=1
    for r,(_,te) in enumerate(skf.split(idx,y)):
        M2[r,te]=1

    outer=StratifiedKFold(4,shuffle=True,random_state=42)
    outer_train,outer_test=next(outer.split(idx,y))
    inner=StratifiedKFold(5,shuffle=True,random_state=42)
    M3=np.zeros((6,n),dtype=int)
    M3[0,outer_test]=2
    for r,(tr_rel,val_rel) in enumerate(inner.split(outer_train,y[outer_train]),start=1):
        M3[r,outer_test]=2
        M3[r,outer_train[val_rel]]=1

    cmap=ListedColormap(["#ECEFF1",PALETTE[1],PALETTE[2]])
    norm=BoundaryNorm([-.5,.5,1.5,2.5],cmap.N)
    fig,axs=plt.subplots(1,3,figsize=(10.4,3.15))
    for ax,M in zip(axs,[M1,M2,M3]):
        ax.imshow(M,aspect="auto",cmap=cmap,norm=norm,interpolation="nearest")
        ax.set_xlabel("Sample index")
        ax.set_xticks([0,12,24,36,47])
    axs[0].set_ylabel("Fold")
    axs[0].set_yticks(range(6),[str(i) for i in range(1,7)])
    axs[1].set_yticks(range(6),[str(i) for i in range(1,7)])
    axs[2].set_yticks(range(6),["outer test","inner 1","inner 2","inner 3","inner 4","inner 5"])
    axs[2].legend(handles=[
        Patch(facecolor="#ECEFF1",edgecolor="none",label="training"),
        Patch(facecolor=PALETTE[1],edgecolor="none",label="validation/test"),
        Patch(facecolor=PALETTE[2],edgecolor="none",label="outer test")
    ],frameon=False,fontsize=7,loc="upper right")
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_13_cross_validation")

def fig_0117_clustering():
    X,_=make_blobs(n_samples=500,centers=4,cluster_std=[.55,.75,.45,.8],random_state=42)
    algs=[
      KMeans(4,n_init=20,random_state=42).fit_predict(X),
      AgglomerativeClustering(4).fit_predict(X),
      DBSCAN(eps=.55,min_samples=8).fit_predict(X),
      GaussianMixture(4,random_state=42).fit_predict(X)
    ]
    fig,axs=plt.subplots(1,4,figsize=(12,2.8))
    for ax,l in zip(axs,algs): ax.scatter(X[:,0],X[:,1],c=l,cmap="tab10",s=8,linewidths=0); ax.set_xticks([]); ax.set_yticks([])
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_17_clustering")

def _surface(ax,model,X,y):
    xx,yy=np.meshgrid(np.linspace(X[:,0].min()-.6,X[:,0].max()+.6,250),np.linspace(X[:,1].min()-.6,X[:,1].max()+.6,250))
    z=model.decision_function(np.c_[xx.ravel(),yy.ravel()]) if hasattr(model,"decision_function") else model.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
    z=z.reshape(xx.shape); ax.contourf(xx,yy,z,levels=15,cmap="RdBu",alpha=.18); ax.contour(xx,yy,z,levels=[0] if z.min()<0<z.max() else [.5],colors="#444",linewidths=1)
    ax.scatter(X[:,0],X[:,1],c=y,cmap="coolwarm",s=15,edgecolor="white",linewidth=.3)

def fig_0118_svm():
    X,y=make_moons(n_samples=350,noise=.18,random_state=42)
    models=[SVC(kernel="linear",C=1).fit(X,y),SVC(kernel="rbf",C=2,gamma=2).fit(X,y)]
    fig,axs=plt.subplots(1,2,figsize=(8.3,3.3))
    for ax,m in zip(axs,models):
        _surface(ax,m,X,y); sv=m.support_vectors_; ax.scatter(sv[:,0],sv[:,1],s=65,facecolors="none",edgecolors="#222",linewidths=.8)
        ax.set_xticks([]); ax.set_yticks([])
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_18_svm")

def fig_0119_tree():
    X,y=make_moons(n_samples=500,noise=.24,random_state=42)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.35,stratify=y,random_state=42)
    m=DecisionTreeClassifier(max_depth=4,min_samples_leaf=10,random_state=42).fit(Xtr,ytr)
    fig,axs=plt.subplots(1,2,figsize=(8.8,3.45))
    _surface(axs[0],m,Xte,yte)
    axs[0].set_xticks([]); axs[0].set_yticks([])

    depths=np.arange(1,11)
    train_scores=[]; test_scores=[]; leaves=[]
    for d in depths:
        t=DecisionTreeClassifier(max_depth=int(d),min_samples_leaf=2,random_state=42).fit(Xtr,ytr)
        train_scores.append(balanced_accuracy_score(ytr,t.predict(Xtr)))
        test_scores.append(balanced_accuracy_score(yte,t.predict(Xte)))
        leaves.append(t.get_n_leaves())
    axs[1].plot(depths,train_scores,marker="o",ms=3.5,lw=1.8,color=PALETTE[0],label="training")
    axs[1].plot(depths,test_scores,marker="o",ms=3.5,lw=1.8,color=PALETTE[1],label="validation")
    axs[1].set_xlabel("Maximum tree depth"); axs[1].set_ylabel("Balanced accuracy"); axs[1].set_ylim(.5,1.02)
    ax2=axs[1].twinx()
    ax2.plot(depths,leaves,ls="--",lw=1.5,color=PALETTE[2],label="leaf count")
    ax2.set_ylabel("Leaf count")
    h1,l1=axs[1].get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    axs[1].legend(h1+h2,l1+l2,frameon=False,fontsize=8,loc="lower right")
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_19_tree")

def fig_0120_ensembles():
    X,y=make_moons(n_samples=450,noise=.25,random_state=42)
    base=DecisionTreeClassifier(max_depth=3,random_state=42)
    models=[
      BaggingClassifier(estimator=base,n_estimators=80,random_state=42).fit(X,y),
      GradientBoostingClassifier(n_estimators=80,max_depth=2,random_state=42).fit(X,y),
      RandomForestClassifier(n_estimators=120,max_depth=5,random_state=42).fit(X,y),
      StackingClassifier(estimators=[("rf",RandomForestClassifier(n_estimators=60,max_depth=4,random_state=42)),("svm",SVC(probability=True,C=2,gamma=2,random_state=42))],final_estimator=LogisticRegression()).fit(X,y)
    ]
    fig,axs=plt.subplots(1,4,figsize=(12,2.8))
    for ax,m in zip(axs,models): _surface(ax,m,X,y); ax.set_xticks([]); ax.set_yticks([])
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_20_ensembles")

def fig_0125_medical():
    img=data.immunohistochemistry(); g=color.rgb2gray(img); thr=filters.threshold_otsu(g); mask=g<thr
    fig,axs=plt.subplots(1,3,figsize=(9.0,3.0))
    axs[0].imshow(img); axs[1].imshow(g,cmap="gray"); axs[2].imshow(mask,cmap="gray")
    for a in axs: a.axis("off")
    for i,a in enumerate(axs): panel(a,f"({chr(97+i)})")
    save(fig,"figure_01_25_medical_preprocessing")

def main():
    global OUT
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args(); OUT=Path(args.out); OUT.mkdir(parents=True,exist_ok=True)
    funcs=[fig_0103_model_capacity,fig_0104_inductive_bias,fig_0106_learning_curves,fig_0107_bias_variance,
           fig_0110_dimensionality,fig_0111_hpo,fig_0112_metrics,fig_0113_cv,fig_0117_clustering,
           fig_0118_svm,fig_0119_tree,fig_0120_ensembles,fig_0125_medical]
    for f in funcs: f()
    manifest={"generated":[f.__name__ for f in funcs],"seed":42,
              "data_sources":["sklearn breast cancer","sklearn digits","sklearn synthetic benchmark datasets","skimage immunohistochemistry sample"]}
    (OUT/"chapter01_foundational_figures_provenance.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__":
    main()
