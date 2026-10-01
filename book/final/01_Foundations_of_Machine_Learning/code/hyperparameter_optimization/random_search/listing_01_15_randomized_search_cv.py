"""Listing 1.15 — Randomized hyperparameter search with cross-validation."""
from __future__ import annotations
from scipy.stats import loguniform, randint
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

def main():
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    model=RandomForestClassifier(random_state=42,n_jobs=-1)
    grid={"n_estimators":randint(80,250),"max_depth":randint(3,18),
          "min_samples_leaf":randint(1,8),"max_features":loguniform(.15,1.0)}
    search=RandomizedSearchCV(model,grid,n_iter=20,cv=5,scoring="roc_auc",
                              random_state=42,n_jobs=-1,refit=True)
    search.fit(Xtr,ytr)
    p=search.best_estimator_.predict_proba(Xte)[:,1]
    print({"best_params":search.best_params_,"cv_auc":search.best_score_,
           "test_auc":roc_auc_score(yte,p)})

if __name__=="__main__":
    main()
