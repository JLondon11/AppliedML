"""Audit Pass-12 generated files without asserting scientific certification.

Usage:
    python audit_pass12_evidence.py --outputs figure_generation/pass12/outputs
    python audit_pass12_evidence.py --outputs figure_generation/pass12/outputs --report audit.json

Exit 0: mechanical evidence checks pass; exit 1: one or more checks fail.
This does NOT validate the current manuscript figure numbering or scientific claims.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def audit(outputs: Path) -> dict:
    checks = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(condition), "detail": detail})

    def metrics(path: Path):
        if not path.is_file():
            check(str(path), False, "Missing metrics JSON")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Metrics JSON root must be an object")
            check(str(path), True, "Readable metrics JSON")
            return data
        except (ValueError, UnicodeError) as exc:
            check(str(path), False, str(exc))
            return None

    def count_check(name: str, data, expected: dict) -> None:
        if data is None:
            check(name, False, "Metrics unavailable")
            return
        observed = {key: data.get(key) for key in expected}
        check(name, observed == expected, f"expected={expected}; observed={observed}")

    def png(name: str, relative: str) -> None:
        p = outputs / relative
        valid = p.is_file() and p.stat().st_size > 24
        if valid:
            with p.open("rb") as stream:
                valid = stream.read(8) == b"\x89PNG\r\n\x1a\n"
        check(name, valid, str(p))

    vae = metrics(outputs / "metrics.json")
    count_check("VAE sample counts", vae, {
        "dataset_samples": 1797, "train_samples": 1347,
        "validation_samples": 450, "test_samples": 0,
    })
    if vae is not None:
        check("VAE evidence limited to validation",
              vae.get("evidence_type") == "measured_validation_not_test",
              str(vae.get("evidence_type")))
        vals = [vae.get("final_train_elbo"),
                vae.get("final_validation_elbo_stochastic"),
                vae.get("validation_latent_10nn_purity_mean")]
        check("VAE finite measured metrics",
              all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals),
              str(vals))

    replacement = metrics(outputs / "replacement_metrics.json")
    count_check("Denoising sample counts", replacement, {
        "training_count": 1347, "validation_count": 450, "test_count": 0,
    })
    if replacement is not None:
        check("Denoising measured noise conditions",
              isinstance(replacement.get("denoising_validation"), list)
              and len(replacement["denoising_validation"]) >= 2,
              "At least two measured validation conditions required")

    ddpm = metrics(outputs / "ddpm" / "metrics.json")
    count_check("DDPM train/validation/test counts", ddpm, {
        "dataset_count": 1797, "training_count": 1257,
        "validation_count": 270, "test_count": 270,
    })
    if ddpm is not None:
        scores = ddpm.get("independent_test_mse")
        check("DDPM independent test scores",
              isinstance(scores, dict) and len(scores) >= 2
              and all(isinstance(v, dict)
                      and all(isinstance(v.get(k), (int, float))
                              and math.isfinite(v[k]) and v[k] >= 0
                              for k in ("predicted_noise_mse", "zero_predictor_mse"))
                      for v in scores.values()),
              "Numeric model and reference error required for multiple timesteps")

    for number in (3, 10, 11, 14, 20, 24):
        png(f"Figure {number:02d} PNG", f"figure_{number:02d}.png")
    for name in ("training_history.csv", "test_metrics.csv",
                 "ddpm_generated_samples.png", "real_test_digits.png"):
        path = outputs / "ddpm" / name
        check(f"DDPM artifact {name}",
              path.is_file() and path.stat().st_size > 0, str(path))

    return {
        "scope": "Pass-12 output file checks only; not manuscript or unit certification",
        "passed": sum(c["pass"] for c in checks),
        "failed": sum(not c["pass"] for c in checks),
        "checks": checks,
        "outstanding_human_review": [
            "Reconcile Figure 20 digit experiment versus industrial-defect manuscript placement",
            "Inspect actual visual quality, labeling, captions and current print figure numbering",
            "Validate execution environment, data license/provenance, and all scientific claims",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.outputs)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
