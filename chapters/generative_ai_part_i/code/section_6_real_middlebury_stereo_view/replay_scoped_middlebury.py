"""Execute and validate the real Middlebury within-scene CVAE experiment.

The complete model/training/figure-generation implementation is in run.py
in this same section. This entry point performs the 220-step full real-data
replay and checks original source hashes, split sizes, 225 scored test patches,
four independently aggregated metric means, and the artifact inventory.
No synthetic fixture images or simulated experimental scores are used.
Usage: python replay_scoped_middlebury.py --output /path/to/results
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from skimage import data

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "mean_test_PSNR_copy_left": 23.157011094530336,
    "mean_test_SSIM_copy_left": 0.5400137761007581,
    "mean_test_PSNR_CVAE_prior_mean": 14.408218237789152,
    "mean_test_SSIM_CVAE_prior_mean": 0.6198464148243268,
}
SOURCE = {
    "left": "ca829467c1d4f427da9c4862ba43829da6ac90afe1f75735e95dba9e3fd9620b",
    "right": "ae44d83f55e66623c7985499fd2f1685a56023e442e66eca89b3457dd46b17af",
}

def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    left, right, _ = data.stereo_motorcycle()
    for name, image in (("left", left), ("right", right)):
        actual = hashlib.sha256(image.tobytes()).hexdigest()
        if actual != SOURCE[name] or manifest["source_sha256_" + name] != actual:
            raise AssertionError("Original real stereo photograph SHA mismatch: " + name)
    counts = (manifest["n_training_patches"],
              manifest["n_validation_patches"], manifest["n_test_patches"])
    if counts != (855, 180, 225) or manifest["n_parent_stereo_pairs"] != 1:
        raise AssertionError("Unsupported partition or source-scene count")
    if manifest["train_rows"][1] >= manifest["validation_rows"][0]:
        raise AssertionError("Training/validation region overlap")
    if manifest["validation_rows"][1] >= manifest["test_rows"][0]:
        raise AssertionError("Validation/test region overlap")
    if manifest["steps"] != 220:
        raise AssertionError("Expected full 220-step replay")
    with (output / "test_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 225 or sorted(int(q["test_patch"]) for q in rows) != list(range(225)):
        raise AssertionError("Incomplete or duplicate held-out sample records")
    pairs = [
        ("PSNR_copy_left", "mean_test_PSNR_copy_left"),
        ("SSIM_copy_left", "mean_test_SSIM_copy_left"),
        ("PSNR_CVAE_prior_mean", "mean_test_PSNR_CVAE_prior_mean"),
        ("SSIM_CVAE_prior_mean", "mean_test_SSIM_CVAE_prior_mean"),
    ]
    measured = {}
    for field, reference in pairs:
        mean = sum(float(q[field]) for q in rows) / len(rows)
        if not math.isfinite(mean) or abs(mean - manifest[reference]) > 1e-8:
            raise AssertionError("Frozen individual measurements mismatch: " + field)
        if abs(mean - EXPECTED[reference]) > 5e-5:
            raise AssertionError("Unexpected rerun outcome: " + field)
        measured[reference] = mean
    wins_psnr = sum(float(q["PSNR_CVAE_prior_mean"]) >
                    float(q["PSNR_copy_left"]) for q in rows)
    wins_ssim = sum(float(q["SSIM_CVAE_prior_mean"]) >
                    float(q["SSIM_copy_left"]) for q in rows)
    if (wins_psnr, wins_ssim) != (46, 188):
        raise AssertionError("Unexpected held-out comparison")
    with (output / "training.csv").open(newline="") as handle:
        train = list(csv.DictReader(handle))
    if len(train) != 22 or int(train[-1]["step"]) != 220:
        raise AssertionError("Incomplete training trace")
    for file in ("conditional_stereo_vae.pt", "figures/stereo_real_model_results.pdf",
                 "figures/stereo_paired_metrics.pdf"):
        if (output / file).stat().st_size < 10000:
            raise AssertionError("Expected executed checkpoint or scientific figure: " + file)
    return {"status": "PASS_REAL_DATA_NARROW_SCOPE",
            "real_stereo_parent_scenes": 1,
            "train_patches": 855, "validation_patches": 180,
            "test_patches": 225, "psnr_cvae_wins": wins_psnr,
            "ssim_cvae_wins": wins_ssim, **measured,
            "scope": "one-scene spatially held-out conditional view prediction only"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        subprocess.run([sys.executable, str(ROOT / "run.py"),
                        "--steps", "220", "--output", str(args.output)], check=True)
    print(json.dumps(verify(args.output), indent=2))

if __name__ == "__main__":
    main()
