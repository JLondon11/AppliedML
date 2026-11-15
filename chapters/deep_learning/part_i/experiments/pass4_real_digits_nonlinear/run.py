"""Chapter 2 Figure 2.5 — genuine handwritten 3-vs-8 nonlinear classification.

Inputs are original Digits photographs, not make_moons or synthetic examples.
StandardScaler and PCA are fitted on training data only. One predefined
32-32 MLP is compared with matched logistic regression on the same 108
held-out original images. No tuning on test labels.
Run python run.py --out outputs
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
NAVY="#203e56";TEAL="#267568";COPPER="#bd7750";SLATE="#8295a2"
def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ds=load_digits();source=ds.data.astype("float64")/16.
    ids=np.where(np.isin(ds.target,[3,8]))[0]
    train,test=train_test_split(ids,test_size=.3,stratify=ds.target[ids],random_state=1729)
    scaler=StandardScaler().fit(source[train])
    pca=PCA(n_components=2,svd_solver="full").fit(scaler.transform(source[train]))
    Xtrain=pca.transform(scaler.transform(source[train]))
    Xtest=pca.transform(scaler.transform(source[test]))
    ytrain=(ds.target[train]==8).astype(int)
    ytest=(ds.target[test]==8).astype(int)
    mlp=MLPClassifier(hidden_layer_sizes=(32,32),max_iter=450,
                      early_stopping=False,random_state=1729).fit(Xtrain,ytrain)
    linear=LogisticRegression(max_iter=1000,random_state=1729).fit(Xtrain,ytrain)
    pred=mlp.predict(Xtest);base=linear.predict(Xtest)
    prob=mlp.predict_proba(Xtest)[:,1]
    score=float(accuracy_score(ytest,pred))
    baseline=float(accuracy_score(ytest,base))
    auc=float(roc_auc_score(ytest,prob))
    gx=np.linspace(min(Xtrain[:,0].min(),Xtest[:,0].min())-1,
                   max(Xtrain[:,0].max(),Xtest[:,0].max())+1,200)
    gy=np.linspace(min(Xtrain[:,1].min(),Xtest[:,1].min())-1,
                   max(Xtrain[:,1].max(),Xtest[:,1].max())+1,200)
    xx,yy=np.meshgrid(gx,gy)
    zz=mlp.predict_proba(np.column_stack([xx.ravel(),yy.ravel()]))[:,1].reshape(xx.shape)
    fig=plt.figure(figsize=(11.6,5.25),layout="constrained")
    gs=fig.add_gridspec(1,3,width_ratios=[.85,1.6,1])
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[0,2])
    photo=np.concatenate([ds.images[train[ytrain==0][0]],
                          ds.images[train[ytrain==1][0]]],axis=1)
    a.imshow(photo,cmap="gray_r",vmin=0,vmax=16,interpolation="nearest")
    a.set(title="(a) Original digit photographs",xticks=[],yticks=[])
    a.text(3.5,8.1,"Recorded 3",ha="center",fontsize=9)
    a.text(11.5,8.1,"Recorded 8",ha="center",fontsize=9)
    b.contourf(xx,yy,zz,levels=np.linspace(0,1,16),cmap="BrBG",alpha=.35)
    b.contour(xx,yy,zz,levels=[.5],colors=[NAVY],linewidths=1.8)
    for value,color,digit in [(0,COPPER,"3"),(1,TEAL,"8")]:
        selection=ytest==value
        b.scatter(Xtest[selection,0],Xtest[selection,1],c=color,s=28,
          marker="o" if value==0 else "^",edgecolors="white",linewidths=.3,
          label=f"Test digit {digit} (n={selection.sum()})")
    b.set(title="(b) Learned MLP boundary on real PCA features",
          xlabel="PC1 (training-fit)",ylabel="PC2 (training-fit)")
    b.legend(frameon=False,fontsize=8)
    error=ytest!=pred
    c.scatter(prob[~error],np.zeros((~error).sum()),s=20,c=TEAL,alpha=.5,label="Correct")
    c.scatter(prob[error],np.ones(error.sum()),s=40,c=COPPER,marker="x",label="Errors")
    c.axvline(.5,lw=1,ls="--",c=SLATE)
    c.set(title="(c) Held-out predictions",xlabel="Predicted probability: digit 8",
          ylabel="Outcome",yticks=[0,1],yticklabels=["Correct","Error"],
          ylim=(-.35,1.35),xlim=(-.05,1.05))
    c.legend(frameon=False,fontsize=8)
    fig.suptitle("Nonlinear classification of original handwritten digit images (3 versus 8)",
      color=NAVY,fontsize=14,fontweight="bold")
    for ext in ("png","pdf","svg"):
        fig.savefig(out/("Figure_2_05_two_moons_boundary."+ext),
                    dpi=235,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    record={
      "dataset":"Original scikit-learn Digits 8x8 handwritten observations",
      "classes":[3,8],"original_records":1797,"subset_records":len(ids),
      "train_records":len(train),"test_records":len(test),
      "train_source_indices":train.tolist(),"test_source_indices":test.tolist(),
      "train_only_transformations":"StandardScaler then PCA(2), trained only on training images",
      "variance_explained":pca.explained_variance_ratio_.tolist(),
      "dataset_sha256_float64":hashlib.sha256(source.tobytes()).hexdigest(),
      "model":"2-feature MLP 32-32, max_iter 450, seed 1729",
      "baseline":"2-feature logistic regression, same held-out cases",
      "iterations":int(mlp.n_iter_),"mlp_accuracy":score,
      "baseline_accuracy":baseline,"mlp_f1":float(f1_score(ytest,pred)),
      "mlp_roc_auc":auc,"test_ground_truth":ytest.tolist(),
      "mlp_test_predict":pred.tolist(),"mlp_test_probabilities":prob.tolist(),
      "baseline_test_predict":base.tolist(),
      "test_embedded_features":Xtest.tolist(),"no_synthetic_images":True,
      "limitations":"Only original digits 3 and 8, two PCA components; single held-out split; no general superiority claim"}
    (out/"real_digits_nonlinear_2d_evidence.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({k:record[k] for k in
       ("subset_records","train_records","test_records","mlp_accuracy",
        "baseline_accuracy","mlp_roc_auc")},indent=2))
    return record
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="outputs")
    run(p.parse_args().out)
