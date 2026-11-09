"""Strict production-grade evidence validation for real-data scientific figures.

A hash proves file integrity, not experimental authenticity; verification therefore
also requires model/dataset identity, a nonempty numeric results record, and a
traceable rendered figure. Visual and scientific review is still a separate gate.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = "chapters/generative_ai_part_ii/"
ALLOWED_SUFFIXES = {".png", ".svg", ".pdf"}
BANNED_GENERATORS = {
    "chapters/_shared/figure_generation_runtime.py",
    "chapters/generative_ai_part_ii/figure_generation/generate_figures.py",
}

def _safe_file(root: Path, relative: object, label: str, issues: list[str]) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        issues.append(f"missing {label} path")
        return None
    if relative.startswith("/") or "\\" in relative:
        issues.append(f"{label} path must be relative POSIX")
        return None
    path = (root / relative).resolve()
    if root not in path.parents:
        issues.append(f"{label} path escapes repository")
        return None
    if not path.is_file():
        issues.append(f"{label} file not committed")
        return None
    return path

def _hash(path: Path | None, value: object, label: str, issues: list[str]) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        issues.append(f"invalid {label} SHA256")
    elif path and hashlib.sha256(path.read_bytes()).hexdigest() != value:
        issues.append(f"{label} SHA256 mismatch")

def _has_measurements(node) -> bool:
    if isinstance(node, dict):
        return any(_has_measurements(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_measurements(v) for v in node)
    return isinstance(node, (float, int)) and not isinstance(node, bool) and math.isfinite(node)

def validate_production(record: dict, root: Path, figure_number: int) -> list[str]:
    root = Path(root).resolve()
    issues = []
    if record.get("status") != "VERIFIED":
        issues.append("figure not verified")
    if record.get("evidence_type") != "measured_real_dataset":
        issues.append("evidence must be measured_real_dataset")
    if record.get("figure_number") != figure_number:
        issues.append("figure number does not match frozen inventory")
    if not isinstance(record.get("dataset_name"), str) or not record["dataset_name"].strip():
        issues.append("missing named real dataset")
    if not isinstance(record.get("dataset_source_url"), str) or not record["dataset_source_url"].startswith(("https://", "http://")):
        issues.append("missing dataset source URL")
    splits = record.get("split_counts")
    if not isinstance(splits, dict) or any(not isinstance(splits.get(k), int) or isinstance(splits.get(k), bool) or splits[k] <= 0 for k in ("train", "validation", "test")):
        issues.append("require positive disjoint train, validation and test counts")
    if record.get("split_disjoint") is not True:
        issues.append("split disjointness not documented")
    if not isinstance(record.get("caption"), str) or len(record["caption"].strip()) < 70:
        issues.append("scientific caption lacks detail")
    if not isinstance(record.get("method"), str) or not record["method"].strip():
        issues.append("missing actual named model or method")
    gen = record.get("generator")
    if gen in BANNED_GENERATORS:
        issues.append("generic fallback generator prohibited")
    source = _safe_file(root, gen, "generator", issues)
    if source and not str(source.relative_to(root)).startswith(ROOT):
        issues.append("generator must be under canonical Chapter 9")
    if source and source.suffix != ".py":
        issues.append("generator must be Python source")
    _hash(source, record.get("generator_sha256"), "generator", issues)
    fig = _safe_file(root, record.get("figure_path"), "figure", issues)
    if fig:
        if not str(fig.relative_to(root)).startswith(ROOT):
            issues.append("figure must be under canonical Chapter 9")
        if fig.suffix.lower() not in ALLOWED_SUFFIXES:
            issues.append("unsupported figure format")
        else:
            data = fig.read_bytes()
            if fig.suffix.lower() == ".png" and not data.startswith(bytes.fromhex("89504e470d0a1a0a")):
                issues.append("invalid PNG header")
            if fig.suffix.lower() == ".pdf" and not data.startswith(b"%PDF-"):
                issues.append("invalid PDF header")
            if fig.suffix.lower() == ".svg" and b"<svg" not in data[:1024]:
                issues.append("invalid SVG header")
    _hash(fig, record.get("figure_sha256"), "figure", issues)
    metrics = _safe_file(root, record.get("metrics_path"), "metrics", issues)
    _hash(metrics, record.get("metrics_sha256"), "metrics", issues)
    if metrics:
        if metrics.suffix != ".json":
            issues.append("metrics must use machine-readable JSON")
        else:
            try:
                values = json.loads(metrics.read_text())
                if not isinstance(values, dict) or not _has_measurements(values):
                    issues.append("metrics contain no measured finite numeric values")
            except (ValueError, UnicodeError, OSError):
                issues.append("metrics JSON unreadable")
    return issues
