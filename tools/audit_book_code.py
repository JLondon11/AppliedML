#!/usr/bin/env python3
"""Repository-wide static reproducibility audit for AppliedML."""
from __future__ import annotations
import argparse
import ast
import csv
import json
import re
from pathlib import Path

RANDOM_RE = re.compile(r"(np\.random|numpy\.random|random\.|torch\.rand|torch\.randn|train_test_split\s*\()")
SEED_RE = re.compile(r"(random\.seed|np\.random\.seed|numpy\.random\.seed|torch\.manual_seed|random_state\s*=|seed\s*=|default_rng\(\s*[^)])")
SANDBOX_RE = re.compile(r"(/mnt/data|/home/oai|sandbox:)")
PLACEHOLDER_RE = re.compile(r"\b(TODO|FIXME)\b", re.I)

def audit(path: Path) -> dict:
    src = path.read_text(encoding="utf-8", errors="replace")
    result = {"path": str(path), "syntax_ok": True, "issues": []}
    try:
        ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        result["syntax_ok"] = False
        result["issues"].append(f"syntax error: {exc}")
    if RANDOM_RE.search(src) and not SEED_RE.search(src):
        result["issues"].append("randomness detected without explicit seed")
    if SANDBOX_RE.search(src):
        result["issues"].append("sandbox-specific hard-coded path")
    if PLACEHOLDER_RE.search(src):
        result["issues"].append("TODO/FIXME token")
    result["status"] = "PASS" if result["syntax_ok"] and not result["issues"] else "REVIEW"
    return result

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="chapters")
    ap.add_argument("--json", default="BOOK_CODE_AUDIT.json")
    ap.add_argument("--csv", default="BOOK_CODE_AUDIT.csv")
    args = ap.parse_args()
    files = sorted(Path(args.root).rglob("*.py"))
    rows = [audit(p) for p in files]
    Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","syntax_ok","status","issues"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "issues": "; ".join(r["issues"])})
    syntax_fail = sum(not r["syntax_ok"] for r in rows)
    review = sum(r["status"] == "REVIEW" for r in rows)
    print(f"Audited {len(rows)} Python files; syntax failures={syntax_fail}; review={review}")
    for r in rows:
        if r["status"] == "REVIEW":
            print("REVIEW: {} :: {}".format(r["path"], "; ".join(r["issues"])))
    return 1 if syntax_fail else 0

if __name__ == "__main__":
    raise SystemExit(main())
