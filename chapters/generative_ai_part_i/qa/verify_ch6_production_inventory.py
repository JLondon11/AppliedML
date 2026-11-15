"""Fail-closed Chapter 6 full production inventory audit.

Scope: requires a current, source-linked inventory of every published figure,
all substantive tables, listings, and 14 parent applications. A passing
inventory is structural evidence only; it never supersedes scientific review.
Run: python verify_ch6_production_inventory.py --repo-root REPO_ROOT
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

CHAPTER = Path("chapters/generative_ai_part_i")
BASELINE_FIGURES = 48  # archived 109-page Pass 44; update only after new compiled audit
BASELINE_PARENTS = 14


def _safe(root: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        return None
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        return None
    full = (root / p).resolve()
    if not full.is_relative_to(root.resolve()) or not full.is_file():
        return None
    return full


def verify(root: Path, inventory: Path | None = None) -> dict:
    root = root.resolve()
    path = inventory or root / CHAPTER / "quality_assurance/production_inventory.json"
    issues: list[str] = []
    if not path.is_file():
        return {"status": "BLOCKED", "issues": ["Missing current full production inventory"], "checks": 0}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"status": "BLOCKED", "issues": [f"Unreadable production inventory: {e}"], "checks": 0}
    if not isinstance(obj, dict):
        return {"status": "BLOCKED", "issues": ["Production inventory must be an object"], "checks": 0}
    if obj.get("chapter") != 6:
        issues.append("Wrong chapter")
    if obj.get("compiled_source_reviewed") is not True:
        issues.append("Compiled manuscript/source not directly reviewed")
    figures = obj.get("figures", [])
    parents = obj.get("parent_applications", [])
    tables = obj.get("tables", [])
    listings = obj.get("code_listings", [])
    if not all(isinstance(v, list) for v in (figures, parents, tables, listings)):
        return {"status": "BLOCKED", "issues": issues + ["Inventories must be arrays"], "checks": 0}
    # Counts must be reconciled to the last independently audited chapter baseline.
    # Revisions changing chapter inventory require a new, documented baseline.
    if len(figures) != BASELINE_FIGURES:
        issues.append(f"Figure coverage {len(figures)}/{BASELINE_FIGURES} audited baseline")
    if len(parents) != BASELINE_PARENTS:
        issues.append(f"Parent applications {len(parents)}/{BASELINE_PARENTS} audited baseline")
    if not isinstance(obj.get("manuscript_sha256"), str) or len(obj["manuscript_sha256"]) != 64:
        issues.append("No current compiled manuscript SHA-256")
    elif (pdf := _safe(root, obj.get("manuscript_pdf"))) is None or hashlib.sha256(pdf.read_bytes()).hexdigest() != obj["manuscript_sha256"]:
        issues.append("Current compiled manuscript file absent or SHA-256 mismatch")
    def unique(rows: list, field: str, category: str) -> set[str]:
        ids = [r.get(field) for r in rows if isinstance(r, dict)]
        if len(ids) != len(rows) or any(not isinstance(v, str) or not v.strip() for v in ids) or len(ids) != len(set(ids)):
            issues.append(f"{category}: missing/duplicate {field}")
        return {v for v in ids if isinstance(v, str)}
    figids = unique(figures, "figure_id", "figures")
    unique(parents, "unit_id", "parent applications")
    unique(tables, "table_id", "tables")
    unique(listings, "listing_id", "code listings")
    for r in figures:
        if not isinstance(r, dict):
            continue
        fid = r.get("figure_id", "?")
        asset = _safe(root, r.get("asset"))
        digest = r.get("sha256")
        if asset is None or not isinstance(digest, str) or hashlib.sha256(asset.read_bytes()).hexdigest() != digest:
            issues.append(f"{fid}: missing figure file or SHA-256 mismatch")
        if not r.get("caption") or r.get("scientific_reviewed") is not True:
            issues.append(f"{fid}: missing caption or scientific figure review")
        if r.get("kind") == "experimental":
            if _safe(root, r.get("generator")) is None or _safe(root, r.get("observed_data_or_results")) is None:
                issues.append(f"{fid}: missing experimental generator or source-derived results")
        elif r.get("kind") != "conceptual":
            issues.append(f"{fid}: must classify experimental versus conceptual")
    for r in parents:
        if not isinstance(r, dict):
            continue
        pid = r.get("unit_id", "?")
        if not r.get("exact_manuscript_title") or r.get("scope_reviewed") is not True:
            issues.append(f"{pid}: title/claims not reviewed")
        if not isinstance(r.get("figure_ids"), list) or not r["figure_ids"] or not set(r["figure_ids"]).issubset(figids):
            issues.append(f"{pid}: missing actual mapped manuscript figure")
        if r.get("scientific_status") not in ("SCIENTIFIC_CERTIFIED", "SCIENTIFIC_HOLD"):
            issues.append(f"{pid}: scientific status required")
        if r.get("scientific_status") == "SCIENTIFIC_CERTIFIED" and not r.get("source_evidence"):
            issues.append(f"{pid}: no referenced scientific evidence")
    for r in tables:
        if not isinstance(r, dict):
            continue
        if r.get("substantive_reviewed") is not True or not r.get("source_or_derivation"):
            issues.append(f"Table {r.get('table_id')}: not justified")
    for r in listings:
        if not isinstance(r, dict):
            continue
        if _safe(root, r.get("complete_code")) is None or r.get("code_reviewed") is not True:
            issues.append(f"Listing {r.get('listing_id')}: incomplete or unchecked source")
    return {"status": "BLOCKED" if issues else "STRUCTURAL_PASS_MANUAL_REVIEW_REQUIRED",
            "issues": issues, "checks": len(figures) + len(parents) + len(tables) + len(listings)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--inventory", type=Path)
    args = p.parse_args()
    result = verify(args.repo_root, args.inventory)
    print(json.dumps(result, indent=2))
    return int(bool(result["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
