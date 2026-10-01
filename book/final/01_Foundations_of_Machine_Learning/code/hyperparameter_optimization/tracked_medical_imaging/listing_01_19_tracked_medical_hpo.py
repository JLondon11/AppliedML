"""Listing 1.19 — Tracked HPO for a reproducible medical-imaging benchmark.

The runnable demonstration uses sklearn digits as a lightweight stand-in for an
image-classification pipeline. Dataset identity and split metadata are logged
explicitly; replace DATASET_ID and the loader with the target medical dataset
(e.g. CheXpert) while preserving patient-level split semantics.
"""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np, optuna
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

DATASET_ID="sklearn_digits_demo"
SPLIT_PROTOCOL="stratified train/validation split; medical use requires patient-level separation"
SEED=42
OUT=Path("tracked_hpo_results")

def objective(trial):
    X,y=load_digits(return_X_y=True)
    Xtr,Xva,ytr,yva=train_test_split(X,y,test_size=.25,stratify=y,random_state=SEED)
    params={
        "n_estimators":trial.suggest_int("n_estimators",80,250),
        "max_depth":trial.suggest_int("max_depth",4,20),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,6),
    }
    model=RandomForestClassifier(**params,random_state=SEED,n_jobs=-1)
    model.fit(Xtr,ytr)
    score=balanced_accuracy_score(yva,model.predict(Xva))
    trial.set_user_attr("dataset",DATASET_ID)
    trial.set_user_attr("split_protocol",SPLIT_PROTOCOL)
    trial.set_user_attr("seed",SEED)
    return float(score)

def main():
    random.seed(SEED); np.random.seed(SEED); OUT.mkdir(exist_ok=True)
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective,n_trials=20)
    payload={
        "dataset":DATASET_ID,"split_protocol":SPLIT_PROTOCOL,"seed":SEED,
        "best_value":study.best_value,"best_params":study.best_params,
        "n_trials":len(study.trials)
    }
    (OUT/"summary.json").write_text(json.dumps(payload,indent=2))
    study.trials_dataframe().to_csv(OUT/"trials.csv",index=False)
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
