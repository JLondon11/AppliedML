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
OUT_SVG = "../figures/figure_08_wdbc_classical_baselines.svg"
OUT_CSV = "../data/figure_08_wdbc_cv_results.csv"

data = load_breast_cancer()
X, y = data.data, data.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=False)
models = {
    "Logistic regression": LogisticRegression(max_iter=5000, solver="liblinear", random_state=SEED),
    "RBF-SVC": SVC(kernel="rbf", probability=True, random_state=SEED),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "k-nearest neighbors": KNeighborsClassifier(n_neighbors=5),
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
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9.5,"axes.linewidth":0.8})
fig, ax = plt.subplots(figsize=(10.8, 6.6))
pos = np.arange(len(models)); offsets = np.linspace(-0.10, 0.10, 5)
for i, name in enumerate(models):
    s = df[df.model == name].sort_values("fold")
    ax.scatter(np.full(5, pos[i])+offsets, s.auroc, s=28, facecolors="white",
               edgecolors="#526778", linewidths=.9, zorder=3)
    mu, sd = s.auroc.mean(), s.auroc.std(ddof=1)
    ax.errorbar(pos[i], mu, yerr=sd, fmt="o", markersize=6.2, capsize=4,
                linewidth=1.25, color="#30343A", zorder=4)
    ax.text(pos[i], mu+sd+.006, f"{mu:.3f}", ha="center", va="bottom", fontsize=8.2)
ax.set_xticks(pos, list(models)); ax.set_ylabel("Cross-validated ROC–AUC")
ax.set_ylim(.88, 1.012); ax.grid(axis="y", linewidth=.55, alpha=.28)
ax.spines[["top","right"]].set_visible(False)
ax.text(0, 1.025,
        f"WDBC: n={len(X)}; 30 features   |   train/test={len(Xtr)}/{len(Xte)} "
        "(stratified, seed 42)   |   5 identical stratified CV folds   |   "
        "StandardScaler fitted within each fold",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, color="#4B5055")
fig.subplots_adjust(bottom=.23, top=.88, left=.10, right=.98)
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
