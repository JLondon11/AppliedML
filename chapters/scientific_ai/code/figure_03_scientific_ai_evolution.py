"""Scientific AI Figure 3: authentic biomedical observations and measured ML outputs.

Uses the original Wisconsin Diagnostic Breast Cancer dataset (UCI, 569 records).
All plotted observations, fitted probabilities, validation scores, and ROC
points derive from actual dataset records and executed logistic regression.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,roc_auc_score,roc_curve,confusion_matrix,accuracy_score

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=load_breast_cancer()
    X=data.data.astype(np.float64)
    y=(data.target==0).astype(int)  # malignant = positive
    assert X.shape==(569,30) and np.array_equal(np.unique(y),[0,1])
    ids=np.arange(len(y))
    trv,te=train_test_split(ids,test_size=114,stratify=y,random_state=20261109)
    tr,va=train_test_split(trv,test_size=114,stratify=y[trv],random_state=20261110)
    assert (len(tr),len(va),len(te))==(341,114,114)
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    scaler=StandardScaler().fit(X[tr])
    xt=scaler.transform(X[tr]);xv=scaler.transform(X[va]);xe=scaler.transform(X[te])
    C_values=[0.01,0.1,1,10,100]
    val_results=[]
    for C in C_values:
        m=LogisticRegression(C=C,max_iter=2000,random_state=20261109).fit(xt,y[tr])
        val_results.append({"C":C,"validation_f1":float(f1_score(y[va],m.predict(xv)))})
    chosen=max(val_results,key=lambda row:row["validation_f1"])["C"]
    model=LogisticRegression(C=chosen,max_iter=2000,random_state=20261109).fit(xt,y[tr])
    prob=model.predict_proba(xe)[:,1]
    pred=(prob>=.5).astype(int)
    auc=float(roc_auc_score(y[te],prob))
    accuracy=float(accuracy_score(y[te],pred))
    f1=float(f1_score(y[te],pred))
    cm=confusion_matrix(y[te],pred,labels=[0,1])
    fpr,tpr,_=roc_curve(y[te],prob)
    assert cm.sum()==114
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig=plt.figure(figsize=(13.2,8.7),layout="constrained")
    gs=fig.add_gridspec(2,6)
    a=fig.add_subplot(gs[0,0:2]);b=fig.add_subplot(gs[0,2:4]);c=fig.add_subplot(gs[0,4:6])
    d=fig.add_subplot(gs[1,0:3]);e=fig.add_subplot(gs[1,3:6])
    for label,color,name in [(0,"#387c71","Benign"),(1,"#b66b43","Malignant")]:
        selected=y[tr]==label
        a.scatter(X[tr,0][selected],X[tr,1][selected],
                  s=16,alpha=.7,color=color,label=name)
    a.set(title="(a) Real cellular measurements",
          xlabel="mean radius",ylabel="mean texture")
    a.legend(frameon=False,fontsize=8)
    corr=np.corrcoef(X[tr,:8],rowvar=False)
    img=b.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1)
    b.set(title="(b) Measured feature correlations",
          xticks=range(8),yticks=range(8))
    b.set_xticklabels([f"F{i+1}" for i in range(8)],rotation=45,fontsize=8)
    b.set_yticklabels([f"F{i+1}" for i in range(8)],fontsize=8)
    fig.colorbar(img,ax=b,fraction=.045,pad=.02,label="Pearson r")
    c.semilogx(C_values,[r["validation_f1"] for r in val_results],
              "-o",color="#315b79",lw=2)
    c.axvline(chosen,ls=":",lw=1.2,color="#b66b43")
    c.set(title="(c) Observed validation performance",
          xlabel="Ridge penalty inverse C",ylabel="Malignancy F1 (validation)",
          ylim=(0,1.03))
    bins=np.linspace(0,1,13)
    d.hist(prob[y[te]==0],bins=bins,alpha=.65,label="Benign test cases",color="#387c71")
    d.hist(prob[y[te]==1],bins=bins,alpha=.65,label="Malignant test cases",color="#b66b43")
    d.set(title="(d) Held-out model outputs (114 patients)",
          xlabel="Predicted malignancy probability",ylabel="Number of held-out cases")
    d.legend(frameon=False,fontsize=9)
    e.plot(fpr,tpr,lw=2,color="#315b79",
           label=f"Measured held-out ROC (AUC {auc:.3f})")
    e.scatter([0],[0],color="#315b79",s=18)
    e.set(title="(e) Held-out discrimination",xlabel="False-positive rate",
          ylabel="True-positive rate",xlim=(0,1),ylim=(0,1.01))
    e.legend(frameon=False,fontsize=9)
    for ax in [a,c,d,e]:ax.grid(alpha=.13)
    for ext in ("svg","png","pdf"):
        fig.savefig(out/f"figure_03_evolution_scientific_ai.{ext}",dpi=220,
                    facecolor="white")
    plt.close(fig)
    result={
      "dataset":"UCI Wisconsin Diagnostic Breast Cancer (scikit-learn)",
      "dataset_url":"https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
      "n_samples":569,"n_features":30,"training":341,"validation":114,"testing":114,
      "partition_seed_train_test":20261109,"partition_seed_train_validation":20261110,
      "source_data_float64_sha256":hashlib.sha256(X.tobytes()).hexdigest(),
      "malignant_is_positive_label":True,
      "feature_names":list(data.feature_names),
      "validation_model_selection":val_results,"chosen_C":chosen,
      "test_accuracy":accuracy,"test_f1_malignant":f1,"test_roc_auc":auc,
      "test_confusion_matrix_true_benign_malignant":cm.tolist(),
      "test_labels":y[te].tolist(),
      "test_probabilities_malignant":prob.tolist(),
      "test_predicted_labels":pred.tolist(),
      "test_roc_fpr":fpr.tolist(),"test_roc_tpr":tpr.tolist(),
      "limitations":"Single stratified split and modest sample; not a clinical deployment evaluation."}
    assert np.isclose(roc_auc_score(y[te],np.asarray(result["test_probabilities_malignant"])),auc)
    (out/"figure_03_scientific_ai_real_data_evidence.json").write_text(
        json.dumps(result,indent=2)+"\n")
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,
                    default=Path(__file__).resolve().parents[1]/"figures")
    args=ap.parse_args()
    result=main(args.output_dir)
    print(json.dumps({k:result[k] for k in
       ["n_samples","training","validation","testing","chosen_C",
        "test_accuracy","test_f1_malignant","test_roc_auc"]},indent=2))
