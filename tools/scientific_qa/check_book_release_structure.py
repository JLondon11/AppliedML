#!/usr/bin/env python3
"""Conservative structural readiness gate; never grants scientific certification."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

CHAPTERS = (
    (1, "Foundations of Machine Learning", "chapters/foundations_machine_learning"),
    (2, "Deep Learning Part I", "chapters/deep_learning/part_i"),
    (3, "Deep Learning Part II", "chapters/deep_learning/part_ii"),
    (4, "Computer Vision Part I", "chapters/computer_vision/part_i/foundations"),
    (5, "Computer Vision Part II", "chapters/computer_vision/part_ii/autonomous_vehicles"),
    (6, "Natural Language Processing Part I", "chapters/nlp_part_i"),
    (7, "Natural Language Processing Part II", "chapters/nlp_part_ii"),
    (8, "Generative AI Part I", "chapters/generative_ai_part_i"),
    (9, "Generative AI Part II", "chapters/generative_ai_part_ii"),
    (10, "Reinforcement Learning", "chapters/reinforcement_learning"),
    (11, "Scientific AI", "chapters/scientific_ai"),
    (12, "Systems Engineering and MLOps", "chapters/systems_engineering_and_mlops"),
)
REQUIRED = {
    "manuscript_latex": ("manuscript", {".tex"}),
    "manuscript_pdf": ("manuscript", {".pdf"}),
    "canonical_figures": ("figures", {".pdf", ".png", ".svg", ".jpg", ".jpeg", ".webp"}),
    "canonical_code": ("code", {".py", ".ipynb", ".r", ".jl", ".m", ".sh"}),
    "instructor_solutions": ("instructor_solutions", {".tex", ".pdf", ".md", ".docx"}),
    "qa_records": ("qa", {".json", ".md", ".csv", ".txt"}),
}

def present(folder: Path, extensions: set[str]) -> bool:
    return folder.is_dir() and any(
        p.is_file() and p.suffix.lower() in extensions and p.stat().st_size > 0
        for p in folder.rglob("*")
    )

def audit(repo_root: Path) -> dict:
    chapters = []
    for number, name, location in CHAPTERS:
        chapter = repo_root / location
        checks = {
            key: present(chapter / subfolder, extensions)
            for key, (subfolder, extensions) in REQUIRED.items()
        }
        checks["release_manifest"] = (chapter / "RELEASE_MANIFEST.md").is_file()
        chapters.append({
            "chapter": number,
            "name": name,
            "root": location,
            "checks": checks,
            "structurally_ready": all(checks.values()),
            "scientifically_certified": False,
            "certification_note": "Not evaluated; structural existence cannot prove empirical or editorial validity.",
        })
    return {
        "schema_version": 1,
        "total_chapters": len(chapters),
        "structurally_ready_chapters": sum(c["structurally_ready"] for c in chapters),
        "book_structurally_ready": all(c["structurally_ready"] for c in chapters),
        "book_scientifically_certified": False,
        "chapters": chapters,
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.repo_root.is_dir():
        parser.error(f"Repository root does not exist: {args.repo_root}")
    report = audit(args.repo_root)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0 if report["book_structurally_ready"] else 1

if __name__ == "__main__":
    sys.exit(main())
