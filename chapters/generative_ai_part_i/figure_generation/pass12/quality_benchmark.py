"""Classifier-feature generated-image assessment on real handwritten digits.

This is not canonical Inception FID; it uses training-only 32-unit digits-MLP
features and reports real/real calibration plus 3 seeded generator runs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from ddpm_digits_experiment import NoisePredictor, sample_ddpm, schedule
from ddim_ddpm_comparison import sample_ddim


def domain_frechet(x, y):
    """PSD-stabilized, biased empirical Fréchet distance in classifier features."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    eig, vec = np.linalg.eigh(cx)
    half = (vec * np.sqrt(np.maximum(eig, 0))) @ vec.T
    cross = half @ cy @ half
    cross = 0.5 * (cross + cross.T)
    trace_root = np.sqrt(np.maximum(np.linalg.eigvalsh(cross), 0)).sum()
    return max(0., float(np.sum((mx-my)**2) + np.trace(cx+cy) - 2*trace_root))


def digit_features(classifier, x):
    # Clipping is explicit and applies ONLY to these downstream classifier features.
    x = np.clip(np.asarray(x, dtype=np.float64), -1, 1)
    z = np.maximum(0, x @ classifier.coefs_[0] + classifier.intercepts_[0])
    return z, classifier.predict(x), classifier.predict_proba(x)


def class_entropy(pred):
    p = np.bincount(pred, minlength=10).astype(float) / len(pred)
    p = p[p > 0]
    return float(-np.sum(p*np.log(p)))


def run(checkpoint, out, samples_per_seed=128, seeds=(201,202,203)):
    if samples_per_seed < 2 or len(seeds) < 2:
        raise ValueError("Require >=2 generated samples and >=2 seeds")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    dataset = load_digits()
    images = dataset.data.astype("float32") / 8 - 1
    labels = dataset.target
    index = np.arange(len(images))
    train_val, test = train_test_split(index, test_size=270, random_state=17,
                                       stratify=labels)
    train, validation = train_test_split(train_val, test_size=270, random_state=17,
                                         stratify=labels[train_val])
    assert (len(train),len(validation),len(test)) == (1257,270,270)
    assert not (set(train)&set(validation) or set(train)&set(test) or set(validation)&set(test))
    classifier = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                               solver="lbfgs", max_iter=240, random_state=23,
                               alpha=0.01)
    classifier.fit(images[train], labels[train])
    validation_accuracy = float(accuracy_score(labels[validation], classifier.predict(images[validation])))
    test_accuracy = float(accuracy_score(labels[test], classifier.predict(images[test])))
    real_feat, _, _ = digit_features(classifier, images[test])
    train_feat, _, _ = digit_features(classifier, images[train])
    neighbor = NearestNeighbors(n_neighbors=1).fit(train_feat)
    real_train_dist = neighbor.kneighbors(real_feat)[0][:, 0]
    near_copy_threshold = float(np.quantile(real_train_dist, .05))
    real_baseline = domain_frechet(real_feat[:128], real_feat[128:256])

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["seed"] == 17 and state["steps"] == 100
    network = NoisePredictor()
    network.load_state_dict(state["model"], strict=True)
    network.eval()
    beta, alpha, alpha_bar = schedule()
    configs = [("DDPM",100),("DDIM",100),("DDIM",50),("DDIM",20)]
    rows = []
    for seed in seeds:
        initial = torch.randn((samples_per_seed,64),
                              generator=torch.Generator().manual_seed(seed))
        for method, steps in configs:
            with torch.no_grad():
                generated = (sample_ddpm(network, samples_per_seed, beta, alpha, alpha_bar, seed)
                             if method == "DDPM" else sample_ddim(network, initial, alpha_bar, steps))
            raw = generated.numpy()
            assert np.isfinite(raw).all()
            features, predicted, probabilities = digit_features(classifier, raw)
            distance = neighbor.kneighbors(features)[0][:,0]
            rows.append({
                "method": method, "steps": steps, "seed": seed,
                "sample_count": len(raw),
                "feature_frechet": domain_frechet(features, real_feat[:len(features)]),
                "mean_max_digit_probability": float(probabilities.max(1).mean()),
                "confident_prediction_fraction": float((probabilities.max(1) >= .8).mean()),
                "predicted_class_entropy": class_entropy(predicted),
                "predicted_classes_present": int(len(np.unique(predicted))),
                "nearest_train_feature_distance_mean": float(distance.mean()),
                "unusually_close_fraction": float((distance < near_copy_threshold).mean()),
                "raw_out_of_range_fraction": float(((raw < -1)|(raw > 1)).mean())
            })
    with (out / "quality_per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    keys = ["feature_frechet","mean_max_digit_probability","predicted_class_entropy",
            "predicted_classes_present","nearest_train_feature_distance_mean",
            "unusually_close_fraction","raw_out_of_range_fraction"]
    for method, steps in configs:
        group = [r for r in rows if r["method"] == method and r["steps"] == steps]
        summary[f"{method}-{steps}"] = {
            key: {"mean": float(np.mean([r[key] for r in group])),
                  "sd_across_seeds": float(np.std([r[key] for r in group], ddof=1))}
            for key in keys}
    evidence = {
        "dataset": "sklearn.datasets.load_digits real 8x8 handwritten digits",
        "split": {"training":1257, "validation":270, "independent_test":270},
        "generated_samples_per_seed": samples_per_seed,
        "seeds": list(seeds),
        "feature_extractor": "training-only sklearn MLPClassifier 32 hidden ReLU units",
        "validation_digit_classification_accuracy": validation_accuracy,
        "independent_test_digit_classification_accuracy": test_accuracy,
        "real_vs_real_feature_frechet_128": real_baseline,
        "near_copy_threshold_test_to_train_feature_p05": near_copy_threshold,
        "metrics": summary,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "limitations": "NOT canonical FID or Inception Score. Classifier confidence is "
                       "not perceptual realism. Range clipping is applied only to "
                       "feature analysis, and all violations are separately reported. "
                       "Near-copy proximity does not prove memorization. This is a "
                       "three-seed small-model descriptive experiment."
    }
    (out / "quality_metrics.json").write_text(json.dumps(evidence, indent=2)+"\n")
    names = list(summary)
    fig, axes = plt.subplots(1,3,figsize=(12.8,3.9))
    for ax,key,title in zip(axes,["feature_frechet","mean_max_digit_probability",
                                   "raw_out_of_range_fraction"],
                             ["Classifier-feature Fréchet (NOT FID)",
                              "Mean maximum digit-class probability",
                              "Raw pixel values outside [-1,1]"]):
        ax.bar(range(4),[summary[n][key]["mean"] for n in names],
               yerr=[summary[n][key]["sd_across_seeds"] for n in names],capsize=3)
        ax.set_xticks(range(4), names, rotation=25)
        ax.set_title(title, fontsize=9)
        if key == "feature_frechet":
            ax.axhline(real_baseline,ls="--",color="black",lw=1,label="real/real reference")
            ax.legend(fontsize=7)
    fig.suptitle("Real-digits trained DDPM/DDIM: 3-seed quality diagnostics")
    fig.tight_layout()
    fig.savefig(out / "quality_diagnostics.png", dpi=190)
    plt.close(fig)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-seed", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint,args.out,args.samples_per_seed),indent=2))
