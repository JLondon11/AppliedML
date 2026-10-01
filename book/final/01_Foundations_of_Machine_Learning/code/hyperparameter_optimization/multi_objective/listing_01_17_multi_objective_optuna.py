"""Listing 1.17 — Multi-objective hyperparameter search with competing directions."""
from __future__ import annotations
import time, optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X,Y=load_breast_cancer(return_X_y=True)
XTR,XVA,YTR,YVA=train_test_split(X,Y,test_size=.25,stratify=Y,random_state=42)

def objective(trial):
    depth=trial.suggest_int("max_depth",3,18)
    trees=trial.suggest_int("n_estimators",50,250)
    model=RandomForestClassifier(n_estimators=trees,max_depth=depth,random_state=42,n_jobs=-1)
    t0=time.perf_counter(); model.fit(XTR,YTR)
    latency=(time.perf_counter()-t0)/len(XTR)
    auc=roc_auc_score(YVA,model.predict_proba(XVA)[:,1])
    return float(auc),float(latency)

if __name__=="__main__":
    study=optuna.create_study(directions=["maximize","minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42))
    study.optimize(objective,n_trials=30)
    for t in study.best_trials[:10]:
        print(t.values,t.params)
