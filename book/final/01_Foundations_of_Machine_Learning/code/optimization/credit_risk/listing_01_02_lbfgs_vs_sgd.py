"""Listing 1.2 — Comparing L-BFGS and SGD-family solvers."""
from __future__ import annotations
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

def compare_solvers(X_train,y_train,X_val,y_val,C=1.0):
    results={}
    for solver in ("lbfgs","sag","saga"):
        clf=make_pipeline(
            StandardScaler(),
            LogisticRegression(solver=solver,C=C,max_iter=2000,tol=1e-6,random_state=42)
        )
        clf.fit(X_train,y_train)
        p=clf.predict_proba(X_val)[:,1]
        lr=clf.named_steps["logisticregression"]
        results[solver]={
            "iterations":int(lr.n_iter_[0]),
            "log_loss":float(log_loss(y_val,p)),
            "roc_auc":float(roc_auc_score(y_val,p)),
        }
    return results

def main():
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    for k,v in compare_solvers(Xtr,ytr,Xv,yv).items():
        print(k,v)

if __name__=="__main__":
    main()
