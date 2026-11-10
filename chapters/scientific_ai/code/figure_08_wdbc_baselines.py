"""Reproduce Scientific AI Figure 8: WDBC classical baseline comparison."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42
from pathlib import Path
import json, hashlib
OUT=Path(__file__).resolve().parent/"outputs"
OUT.mkdir(parents=True,exist_ok=True)
OUT_SVG=OUT/"figure_08_wdbc_classical_baselines.svg"
OUT_CSV=OUT/"cv_fold_results.csv"

data = load_breast_cancer()
X, y = data.data, data.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "Logistic regression": LogisticRegression(max_iter=5000, solver="liblinear", random_state=SEED),
    "RBF-SVC": SVC(kernel="rbf", probability=True, random_state=SEED),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
    "5-NN": KNeighborsClassifier(n_neighbors=5),
}
rows = []
folds = list(cv.split(Xtr, ytr))
for name, est in models.items():
    pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
    for fold, (a, b) in enumerate(folds, 1):
        pipe.fit(Xtr[a], ytr[a])
        p = pipe.predict_proba(Xtr[b])[:, 1]
        q = pipe.predict(Xtr[b])
        rows.append(dict(model=name, fold=fold,
                         auroc=roc_auc_score(ytr[b], p),
                         accuracy=accuracy_score(ytr[b], q)))

df=pd.DataFrame(rows)
df.to_csv(OUT_CSV,index=False)
test_records=[]
test_predictions=[]
curves={}
for name, est in models.items():
    pipeline=Pipeline([("scale",StandardScaler()),("model",est)])
    pipeline.fit(Xtr,ytr)
    proba=pipeline.predict_proba(Xte)[:,1]
    pred=(proba>=0.5).astype(int)
    from sklearn.metrics import roc_curve,confusion_matrix
    fpr,tpr,_=roc_curve(yte,proba)
    curves[name]={"fpr":fpr.tolist(),"tpr":tpr.tolist()}
    test_records.append({"model":name,"test_accuracy":float(accuracy_score(yte,pred)),
                         "test_auroc":float(roc_auc_score(yte,proba)),
                         "confusion_matrix":confusion_matrix(yte,pred,labels=[0,1]).tolist()})
    for i in range(len(yte)):
        test_predictions.append({"model":name,"test_row":i,"true_label":int(yte[i]),
            "predicted_label":int(pred[i]),"probability_benign":float(proba[i])})
pd.DataFrame(test_records).to_csv(OUT/"heldout_test_results.csv",index=False)
pd.DataFrame(test_predictions).to_csv(OUT/"heldout_predictions.csv",index=False)
summary={"dataset":"UCI Wisconsin Diagnostic Breast Cancer","observations":569,
 "feature_count":30,"training":len(Xtr),"heldout_testing":len(Xte),
 "cross_validation_folds":5,"cv_within_training_only":True,
 "dataset_sha256_float64":hashlib.sha256(X.astype("float64").tobytes()).hexdigest(),
 "fold_results":rows,"heldout_metrics":test_records,"roc_curves":curves}
(OUT/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
palette=["#315A79","#C07D53","#397B72","#8A769B"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
  "axes.spines.top":False,"axes.spines.right":False})
fig,axs=plt.subplots(1,2,figsize=(11.8,4.6),layout="constrained")
ax=axs[0]
for i,(name,color) in enumerate(zip(models,palette)):
    s=df[df.model==name].sort_values("fold")
    values=s.auroc.to_numpy()
    ax.scatter(i+np.linspace(-0.15,0.15,5),values,s=30,color=color,alpha=.7)
    ax.errorbar(i,np.mean(values),yerr=np.std(values,ddof=1),
                fmt="D",markersize=6,capsize=5,color="#1E3449",zorder=4)
ax.set_xticks(range(4),["Logistic\nregression","RBF\nSVC","Random\nforest","5-NN"])
ax.set(ylabel="Internal cross-validation ROC AUC",title="(a) Observed fold variability (5 folds)",ylim=(.85,1.015))
ax.grid(axis="y",alpha=.18)
ax=axs[1]
for name,color in zip(models,palette):
    curve=curves[name]
    auc=next(x["test_auroc"] for x in test_records if x["model"]==name)
    ax.plot(curve["fpr"],curve["tpr"],color=color,lw=1.8,label=f"{name}: AUC {auc:.3f}")
ax.plot([0,1],[0,1],ls=":",c="#9FA9B1",lw=.9)
ax.set(xlabel="False-positive rate",ylabel="True-positive rate",
       title="(b) Held-out ROC (114 real cases)",xlim=(0,1),ylim=(0,1.02))
ax.legend(frameon=False,fontsize=8,loc="lower right")
ax.grid(alpha=.16)
for ext in ("svg","png","pdf"):
    fig.savefig(OUT/f"figure_08_wdbc_classical_baselines.{ext}",dpi=240,facecolor="white")
plt.close(fig)
for rec in test_records:
    group=[v for v in test_predictions if v["model"]==rec["model"]]
    assert np.isclose(accuracy_score([v["true_label"] for v in group],
                                      [v["predicted_label"] for v in group]),rec["test_accuracy"])
    assert np.isclose(roc_auc_score([v["true_label"] for v in group],
                                     [v["probability_benign"] for v in group]),rec["test_auroc"])
