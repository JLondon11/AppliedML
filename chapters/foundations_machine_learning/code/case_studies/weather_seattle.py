"""Real-data weather forecasting case study for Chapter 1.

Dataset: Seattle daily weather observations from the Vega datasets repository.
Task: predict next-day precipitation occurrence and amount using lagged observations.
Models: logistic regression and random-forest regressor.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1729
URL = "https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv"

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for c in ["precipitation", "temp_max", "temp_min", "wind"]:
        for lag in [1, 2, 3, 7]:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["target_precip"] = df["precipitation"].shift(-1)
    df["target_rain"] = (df["target_precip"] > 0).astype(int)
    return df.dropna().reset_index(drop=True)

def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(URL)
    data = make_features(raw)
    split = int(0.8 * len(data))
    train, test = data.iloc[:split], data.iloc[split:]
    feats = [c for c in data.columns if "_lag" in c] + ["month"]

    clf = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, random_state=SEED))
    ])
    clf.fit(train[feats], train["target_rain"])
    prob = clf.predict_proba(test[feats])[:, 1]
    pred = (prob >= 0.5).astype(int)

    reg = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3,
        random_state=SEED, n_jobs=-1
    )
    reg.fit(train[feats], train["target_precip"])
    amount = np.maximum(0.0, reg.predict(test[feats]))

    metrics = pd.DataFrame([{
        "rain_accuracy": accuracy_score(test["target_rain"], pred),
        "rain_f1": f1_score(test["target_rain"], pred),
        "rain_roc_auc": roc_auc_score(test["target_rain"], prob),
        "precip_mae_mm": mean_absolute_error(test["target_precip"], amount),
        "precip_rmse_mm": mean_squared_error(test["target_precip"], amount) ** 0.5,
    }])
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    pd.DataFrame({
        "date": test["date"],
        "true_precipitation_mm": test["target_precip"],
        "predicted_precipitation_mm": amount,
        "true_rain": test["target_rain"],
        "predicted_rain_probability": prob,
    }).to_csv(output_dir / "predictions.csv", index=False)

    fpr, tpr, _ = roc_curve(test["target_rain"], prob)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(test['target_rain'], prob):.3f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "weather_rain_roc.svg", bbox_inches="tight")
    fig.savefig(output_dir / "weather_rain_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    n = min(120, len(test))
    x = pd.to_datetime(test["date"].iloc[:n])
    ax.plot(x, test["target_precip"].iloc[:n], label="observed")
    ax.plot(x, amount[:n], label="predicted")
    ax.set_ylabel("Next-day precipitation (mm)")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "weather_precipitation_forecast.svg", bbox_inches="tight")
    fig.savefig(output_dir / "weather_precipitation_forecast.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "dataset": "Seattle Weather",
        "source_url": URL,
        "task": "chronological one-day-ahead precipitation classification and regression",
        "split": "first 80% train / final 20% test",
        "seed": SEED,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("results/weather_seattle"))
    run(p.parse_args().output_dir)
