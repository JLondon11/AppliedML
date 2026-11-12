"""Independent checks of stored Chapter 6 VAE/PCA evidence.

Usage: python audit_vae_pca_evidence.py --results OUTPUT_DIR --report QA.json
Requires numpy and Pillow. Does not retrain models or certify manuscript figures.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def audit(root):
    metrics = json.loads((root / "metrics.json").read_text())
    with (root / "test_predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (root / "training_history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    checks = []

    def check(label, passed, note):
        checks.append({"check": label, "pass": bool(passed), "detail": str(note)})

    ids = np.array([int(r["source_index"]) for r in rows], dtype=np.int64)
    vae = np.array([float(r["vae_mse"]) for r in rows])
    pca = np.array([float(r["pca_mse"]) for r in rows])
    paired = pca - vae
    check("partitions and record count",
          len(rows) == metrics["test_samples"] == 270
          and len(history) == metrics["epochs_max"] == 35
          and metrics["source_samples"] == 1797
          and metrics["train_samples"] == 1257
          and metrics["validation_samples"] == 270, len(rows))
    check("unique source indices", len(np.unique(ids)) == len(ids)
          and np.all((ids >= 0) & (ids < 1797)), len(np.unique(ids)))
    digest = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
    check("test source-index hash", digest == metrics["test_index_sha256"], digest)
    check("finite nonnegative losses",
          np.isfinite(vae).all() and np.isfinite(pca).all()
          and (vae >= 0).all() and (pca >= 0).all(), "both methods")
    check("paired per-image differences",
          np.allclose(paired, [float(r["pca_minus_vae_mse"]) for r in rows]), "PCA minus VAE")
    for label, values, key in [
        ("VAE test MSE", vae, "vae_test_reconstruction_mse_mean"),
        ("PCA test MSE", pca, "pca_test_reconstruction_mse_mean"),
        ("mean paired delta", paired, "paired_pca_minus_vae_mse_mean"),
    ]:
        check(label, np.isclose(values.mean(), metrics[key], atol=1e-7),
              f"recomputed={values.mean():.10f} reported={metrics[key]:.10f}")
    rng = np.random.default_rng(metrics["random_seed"] + 21)
    draws = rng.integers(0, len(rows), size=(metrics["bootstrap_draws"], len(rows)))
    ci = np.quantile(paired[draws].mean(axis=1), [.025, .975])
    check("paired bootstrap CI", np.allclose(ci, metrics["paired_difference_bootstrap_ci95"],
                                           atol=1e-7), ci.tolist())
    image = root / "real_digits_vae_pca_evaluation.png"
    with Image.open(image) as fig:
        size = fig.size
        fig.verify()
    check("figure image", size[0] >= 1600 and size[1] >= 850, size)
    checkpoint = root / "vae_checkpoint.pt"
    check("checkpoint", checkpoint.exists() and checkpoint.stat().st_size > 1000,
          checkpoint.name)
    return {"scope": "File and numerical consistency only; NOT full scientific certification",
            "passed": sum(c["pass"] for c in checks),
            "failed": sum(not c["pass"] for c in checks),
            "checks": checks,
            "file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(root.iterdir()) if p.is_file()},
            "limitations": ["Training not replayed by this script",
                            "Bootstrap conditions on fitted models",
                            "Manuscript figure identity and numbering not established"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    result = audit(args.results)
    message = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(message)
    print(message)
    raise SystemExit(int(result["failed"] != 0))
