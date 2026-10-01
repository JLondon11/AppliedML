"""Listing 1.16 — Define-by-run hyperparameter search using Optuna."""
from __future__ import annotations
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

X,Y=load_breast_cancer(return_X_y=True)

def objective(trial):
    max_depth=trial.suggest_int("max_depth",3,20)
    min_leaf=trial.suggest_int("min_samples_leaf",1,8)
    max_features=trial.suggest_float("max_features",.2,1.0)
    model=RandomForestClassifier(n_estimators=150,max_depth=max_depth,
        min_samples_leaf=min_leaf,max_features=max_features,random_state=42,n_jobs=-1)
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    return float(cross_val_score(model,X,Y,cv=cv,scoring="roc_auc",n_jobs=-1).mean())

if __name__=="__main__":
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=20)
    print(study.best_value,study.best_params)
