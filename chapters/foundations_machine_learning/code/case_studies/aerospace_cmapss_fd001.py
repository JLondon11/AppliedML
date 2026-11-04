"""Real-data aerospace prognostics case study for Chapter 1.

Dataset: NASA C-MAPSS FD001 public mirror.
Task: remaining useful life (RUL) regression with engine-disjoint holdout.
Model: random forest regressor.
"""
from pathlib import Path
import argparse
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 1729
URL = "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/train_FD001.txt"

def phm08_score(y, p):
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))

def load_fd001():
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), sep=r"\s+", header=None).iloc[:, :26]
    df.columns = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    max_cycle = df.groupby("unit")["cycle"].max()
    df["RUL"] = df["unit"].map(max_cycle) - df["cycle"]
    return df

def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_fd001()
    units = np.array(sorted(df["unit"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(units)
    train_units, test_units = units[:70], units[70:]
    tr = df[df["unit"].isin(train_units)].copy()
    te = df[df["unit"].isin(test_units)].copy()

    features = ["cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    model = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3,
        random_state=SEED, n_jobs=-1
    )
    model.fit(tr[features], tr["RUL"])
    pred = np.maximum(0.0, model.predict(te[features]))

    metrics = pd.DataFrame([{
        "rmse_cycles": mean_squared_error(te["RUL"], pred) ** 0.5,
        "mae_cycles": mean_absolute_error(te["RUL"], pred),
        "phm08_score": phm08_score(te["RUL"].to_numpy(), pred),
        "train_engines": len(train_units),
        "test_engines": len(test_units),
    }])
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame({
        "unit": te["unit"].to_numpy(),
        "cycle": te["cycle"].to_numpy(),
        "true_rul": te["RUL"].to_numpy(),
        "predicted_rul": pred,
    }).to_csv(output_dir / "predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    ax.scatter(te["RUL"], pred, s=6, alpha=0.25)
    lim = max(float(te["RUL"].max()), float(np.max(pred)))
    ax.plot([0, lim], [0, lim], "--", linewidth=1)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    fig.tight_layout()
    fig.savefig(output_dir / "cmapss_rul_scatter.svg", bbox_inches="tight")
    fig.savefig(output_dir / "cmapss_rul_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    unit = int(test_units[0])
    one = te[te["unit"] == unit]
    one_pred = pred[te["unit"].to_numpy() == unit]
    fig, ax = plt.subplots(figsize=(6.7, 4.1))
    ax.plot(one["cycle"], one["RUL"], label="true")
    ax.plot(one["cycle"], one_pred, label="prediction")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Remaining useful life")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cmapss_engine_trajectory.svg", bbox_inches="tight")
    fig.savefig(output_dir / "cmapss_engine_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    (output_dir / "provenance.json").write_text(json.dumps({
        "dataset": "NASA C-MAPSS FD001",
        "source_url": URL,
        "split": "engine-disjoint 70 train / 30 test",
        "model": "RandomForestRegressor",
        "seed": SEED,
    }, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("results/aerospace_cmapss_fd001"))
    run(p.parse_args().output_dir)
