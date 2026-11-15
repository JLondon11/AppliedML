"""Original-data Chapter 2 Pass 2 case-study remediation.

Generate four measured manuscript figures (2.31–2.34) using original
scikit-learn handwritten Digits and Diabetes observations, with matched
original-data classical baselines. No fabricated observations or results.
Command: python run.py --output outputs
Dependencies: numpy scipy scikit-learn matplotlib.
"""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED=1729
NAVY="#223F59";TEAL="#347B73";COPPER="#BA7654";SLATE="#879CAA"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
"axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})
def save(fig,out,stem):
    for e in ("png","pdf","svg"):fig.savefig(out/(stem+"."+e),dpi=235,bbox_inches="tight")
    plt.close(fig)
def digits(out):
    dat=load_digits();X=dat.data.astype("float64")/16;y=dat.target
    ids=np.arange(len(y));tr,te=train_test_split(ids,test_size=.25,stratify=y,random_state=SEED)
    mlp=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),early_stopping=False,max_iter=250,random_state=SEED))
    lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200,random_state=SEED))
    mlp.fit(X[tr],y[tr]);lr.fit(X[tr],y[tr])
    yt=y[te];p=mlp.predict(X[te]);pr=mlp.predict_proba(X[te]);bp=lr.predict(X[te]);loss=mlp.named_steps["mlpclassifier"].loss_curve_
    cm=confusion_matrix(yt,p,labels=np.arange(10))
    result={"dataset":"Original scikit-learn 8×8 Digits; not MNIST",
     "source_url":"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
     "records":len(X),"train":len(tr),"test":len(te),"seed":SEED,
     "train_indices":tr.tolist(),"test_indices":te.tolist(),
     "feature_sha256_float64":hashlib.sha256(X.tobytes()).hexdigest(),
     "architecture":"StandardScaler + 64-32 ReLU MLP, 250 maximum iterations, no internal early stop",
     "iteration_count":int(mlp.named_steps["mlpclassifier"].n_iter_),
     "training_loss":loss,
     "accuracy":float(accuracy_score(yt,p)),
     "macro_f1":float(f1_score(yt,p,average="macro")),
     "ovr_auc":float(roc_auc_score(yt,pr,multi_class="ovr")),
     "baseline_accuracy":float(accuracy_score(yt,bp)),
     "baseline_macro_f1":float(f1_score(yt,bp,average="macro")),
     "true_test":yt.tolist(),"predicted_test":p.tolist(),
     "probabilities_test":pr.tolist(),"baseline_predicted_test":bp.tolist(),
     "confusion_matrix":cm.tolist(),
     "scope":"Single original-data fixed split, no cross-seed superiority claim."}
    fig,ax=plt.subplots(1,2,figsize=(10.6,3.8),layout="constrained")
    ax[0].plot(np.arange(1,len(loss)+1),loss,color=TEAL,lw=2)
    ax[0].set(title="(a) Executed MLP training loss",xlabel="Optimization iteration",ylabel="Cross-entropy")
    acc=[result["baseline_accuracy"],result["accuracy"]]
    ax[1].scatter([0,1],acc,c=[SLATE,COPPER],s=94)
    for i,z in enumerate(acc):ax[1].text(i,z+.002,f"{z:.4f}",ha="center",fontsize=9)
    ax[1].set(title="(b) Same 450 original test images",
              ylabel="Accuracy",xlim=(-.5,1.5),ylim=(.94,1.005),xticks=[0,1])
    ax[1].set_xticklabels(["Logistic baseline","64–32 MLP"])
    for a in ax:a.grid(alpha=.15)
    save(fig,out,"Figure_2_31_digits_training_comparison")
    fig,ax=plt.subplots(1,2,figsize=(10.4,4.05),layout="constrained")
    im=ax[0].imshow(cm,cmap="Blues")
    for i in range(10):
        for j in range(10):
            if i==j or cm[i,j]>0:ax[0].text(j,i,str(cm[i,j]),fontsize=7,ha="center",va="center",color="white" if cm[i,j]>22 else NAVY)
    ax[0].set(title="(a) Observed versus predicted",xlabel="Predicted digit",ylabel="True digit",xticks=range(10),yticks=range(10))
    fig.colorbar(im,ax=ax[0],fraction=.05,pad=.02,label="Actual test count")
    recall=cm.diagonal()/cm.sum(1)
    ax[1].plot(range(10),recall,"o-",c=TEAL)
    ax[1].set(title="(b) Measured class recall",xlabel="Digit",ylabel="Recall",xticks=range(10),ylim=(.85,1.02))
    ax[1].grid(alpha=.15)
    save(fig,out,"Figure_2_32_digits_error_structure")
    confidence=pr.max(1);correct=(p==yt).astype(float)
    bins=np.linspace(0,1,11);conf=[];observed=[];count=[]
    for low,high in zip(bins[:-1],bins[1:]):
        mask=(confidence>=low)&(confidence<(high if high<1 else 1.0000001))
        if mask.any():
            conf.append(float(confidence[mask].mean()))
            observed.append(float(correct[mask].mean()))
            count.append(int(mask.sum()))
    fig,ax=plt.subplots(1,2,figsize=(10.3,4.0),layout="constrained")
    ax[0].plot([0,1],[0,1],"--",c=SLATE,lw=1,label="Perfectly calibrated")
    ax[0].scatter(conf,observed,s=np.array(count)*2.4+30,c=TEAL,alpha=.75,label="Original held-out bins")
    ax[0].set(title="(a) Measured reliability",xlabel="Mean class confidence",ylabel="Empirical correct fraction",xlim=(0,1.02),ylim=(0,1.02))
    ax[0].legend(frameon=False,fontsize=8)
    wrong=np.flatnonzero(p!=yt)
    image=np.zeros((16,32))
    for k,wi in enumerate(wrong[:8]):
        image[(k//4)*8:(k//4+1)*8,(k%4)*8:(k%4+1)*8]=dat.images[te[wi]]
    ax[1].imshow(image,cmap="gray_r",vmin=0,vmax=16)
    ax[1].set(title=f"(b) Genuine errors ({len(wrong)} of 450)",xticks=[],yticks=[])
    for k,wi in enumerate(wrong[:8]):
        ax[1].text((k%4)*8+4,(k//4)*8+7.2,f"{yt[wi]}→{p[wi]}",fontsize=8,
          ha="center",color=COPPER,bbox={"facecolor":"white","alpha":.88,"edgecolor":"none","pad":.6})
    ax[0].grid(alpha=.15)
    save(fig,out,"Figure_2_33_digits_calibration_errors")
    return result
def diabetes(out):
    dat=load_diabetes();X=dat.data.astype("float64");y=dat.target.astype("float64")
    ids=np.arange(len(y));tr,te=train_test_split(ids,test_size=.25,random_state=SEED)
    mlp=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,32),early_stopping=False,max_iter=700,random_state=SEED))
    ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.))
    mlp.fit(X[tr],y[tr]);ridge.fit(X[tr],y[tr])
    truth=y[te];p=mlp.predict(X[te]);q=ridge.predict(X[te])
    result={"dataset":"Original scikit-learn diabetes observational progression dataset",
      "source_url":"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
      "records":len(X),"features":X.shape[1],"train":len(tr),"test":len(te),"seed":SEED,
      "train_indices":tr.tolist(),"test_indices":te.tolist(),
      "feature_sha256_float64":hashlib.sha256(X.tobytes()).hexdigest(),
      "model":"StandardScaler + 64-32 MLP, 700 maximum iterations, no early stopping",
      "baseline":"StandardScaler + Ridge(alpha=1) on same split",
      "iteration_count":int(mlp.named_steps["mlpregressor"].n_iter_),
      "training_loss":mlp.named_steps["mlpregressor"].loss_curve_,
      "rmse":float(np.sqrt(mean_squared_error(truth,p))),"mae":float(mean_absolute_error(truth,p)),
      "r2":float(r2_score(truth,p)),"ridge_rmse":float(np.sqrt(mean_squared_error(truth,q))),
      "ridge_r2":float(r2_score(truth,q)),
      "true_test":truth.tolist(),"predicted_test":p.tolist(),"ridge_predicted_test":q.tolist(),
      "limitations":["Small observational dataset; no clinical use established","MLP reached 700 iterations without convergence; Ridge has lower RMSE on this split"]}
    fig,ax=plt.subplots(1,3,figsize=(12.6,3.7),layout="constrained")
    ax[0].scatter(truth,p,s=19,c=TEAL,alpha=.68,rasterized=True);ax[0].plot([20,350],[20,350],"--",lw=.9,c=SLATE)
    ax[0].set(title="(a) Actual held-out predictions (n=111)",xlabel="Observed progression",ylabel="Predicted progression")
    ax[1].scatter(truth,truth-p,s=16,c=COPPER,alpha=.65,rasterized=True);ax[1].axhline(0,ls="--",color=SLATE,lw=1)
    ax[1].set(title="(b) Actual MLP residuals",xlabel="Observed progression",ylabel="Observed minus predicted")
    for residual,col,label in [(np.abs(truth-p),TEAL,f"MLP RMSE {result['rmse']:.2f}"),(np.abs(truth-q),COPPER,f"Ridge RMSE {result['ridge_rmse']:.2f}")]:
        values=np.sort(residual);ax[2].step(values,np.arange(1,len(values)+1)/len(values),where="post",color=col,lw=1.8,label=label)
    ax[2].set(title="(c) Measured absolute errors",xlabel="Absolute error (score units)",ylabel="Empirical cumulative fraction",ylim=(0,1.02))
    ax[2].legend(frameon=False,fontsize=8)
    for a in ax:a.grid(alpha=.15)
    save(fig,out,"Figure_2_34_diabetes_regression_diagnostics")
    return result
def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    d=digits(output);b=diabetes(output)
    x={"digits":d,"diabetes":b,"python":platform.python_version(),
       "sklearn":sklearn.__version__,"scope":"Real Digits and Diabetes only; original two-moons simulation excluded."}
    (output/"measured_results.json").write_text(json.dumps(x,indent=2)+"\n")
    print(json.dumps({"digits_mlp_accuracy":d["accuracy"],
      "digits_logistic_accuracy":d["baseline_accuracy"],
      "diabetes_mlp_rmse":b["rmse"],"diabetes_ridge_rmse":b["ridge_rmse"]},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="outputs")
    main(p.parse_args().output)
