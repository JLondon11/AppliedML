"""Generate Figure 1.57: fold-level Digits accuracy comparison."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=62)
rows = []
for fold, (tr, te) in enumerate(cv.split(X, y), 1):
    models = {
        "Adam MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,), solver="adam", max_iter=500,
                random_state=62, early_stopping=True, n_iter_no_change=30
            ),
        ),
        "Averaged SGD": make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss", average=True, max_iter=3000,
                tol=1e-4, random_state=62
            ),
        ),
    }
    for name, model in models.items():
        model.fit(X[tr], y[tr])
        rows.append((fold, name, accuracy_score(y[te], model.predict(X[te]))))

df = pd.DataFrame(rows, columns=["fold", "model", "accuracy"])
df.to_csv("figure_1_57_fold_accuracy.csv", index=False)

fig, ax = plt.subplots(figsize=(7.2, 4.3))
for model, grp in df.groupby("model"):
    ax.plot(grp["fold"], grp["accuracy"], marker="o", label=model)
ax.set_xticks(range(1, 6))
ax.set_xlabel("Cross-validation fold")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.92, 0.995)
ax.grid(True, alpha=0.2)
ax.legend(frameon=False)
ax.set_title("Fold-level Digits accuracy")
fig.tight_layout()
fig.savefig("figure_1_57_fold_accuracy.png", dpi=300, bbox_inches="tight")
