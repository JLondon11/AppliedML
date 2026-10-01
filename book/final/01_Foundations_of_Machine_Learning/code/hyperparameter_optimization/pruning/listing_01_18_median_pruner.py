"""Listing 1.18 — Hyperparameter search with median-based trial pruning."""
from __future__ import annotations
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

X,Y=load_breast_cancer(return_X_y=True)
XTR,XVA,YTR,YVA=train_test_split(X,Y,test_size=.25,stratify=Y,random_state=42)
SCALER=StandardScaler().fit(XTR); XTR=SCALER.transform(XTR); XVA=SCALER.transform(XVA)

def objective(trial):
    alpha=trial.suggest_float("alpha",1e-6,1e-2,log=True)
    clf=SGDClassifier(loss="log_loss",alpha=alpha,random_state=42,warm_start=True)
    classes=[0,1]
    for epoch in range(30):
        clf.partial_fit(XTR,YTR,classes=classes)
        auc=roc_auc_score(YVA,clf.predict_proba(XVA)[:,1])
        trial.report(auc,step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(auc)

if __name__=="__main__":
    study=optuna.create_study(direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=30)
    print(study.best_value,study.best_params)
