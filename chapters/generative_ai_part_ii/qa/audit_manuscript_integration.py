"""Audit final Chapter 9 LaTeX-to-figure integration; fail closed.

This is a structural gate. It does NOT certify scientific content or visual quality.
Usage:
 python chapters/generative_ai_part_ii/qa/audit_manuscript_integration.py --root . --report chapter9_manuscript_audit.json --strict
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

CHAPTER = Path("chapters/generative_ai_part_ii")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
CAPTION = re.compile(r"\\caption(?:\[[^]]*\])?\{")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REFERENCE = re.compile(r"\\(?:ref|autoref|cref|Cref|eqref)\{([^}]+)\}")
INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def strip_comments(text: str) -> str:
    # TeX comments start at unescaped percent characters.
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0]
                     for line in text.splitlines())


def read_tex_tree(source: Path, root: Path, visited: set[Path] | None = None):
    visited = set() if visited is None else visited
    resolved = source.resolve()
    if resolved in visited:
        return "", []
    if root.resolve() not in resolved.parents or not resolved.is_file():
        raise ValueError(f"Missing or out-of-scope TeX file: {source}")
    visited.add(resolved)
    text = strip_comments(resolved.read_text(encoding="utf-8"))
    visited_files = [resolved]
    for match in INPUT.finditer(text):
        name = match.group(1)
        child = resolved.parent / (name if name.endswith(".tex") else name + ".tex")
        child_text, child_files = read_tex_tree(child, root, visited)
        text = text.replace(match.group(0), "\n" + child_text + "\n", 1)
        visited_files.extend(child_files)
    return text, visited_files


def figures_in_tex(content: str):
    # A figure must include actual graphics AND a caption within one figure environment.
    results = []
    pattern = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.S)
    for match in pattern.finditer(content):
        body = match.group(1)
        results.append({"graphics": GRAPHIC.findall(body),
                        "has_caption": bool(CAPTION.search(body)),
                        "labels": LABEL.findall(body)})
    return results


def audit(root: Path):
    root = root.resolve()
    chapter = root / CHAPTER
    manuscript = chapter / "manuscript"
    tex_files = sorted(manuscript.rglob("*.tex")) if manuscript.is_dir() else []
    errors = []
    if not tex_files:
        errors.append("Canonical Chapter 9 manuscript/*.tex is missing")
        return {"status": "HOLD", "chapter": 9, "source_files": [],
                "figure_environments": 0, "checked_graphics": 0, "errors": errors}
    roots = sorted([x for x in tex_files if "\\documentclass" in x.read_text(encoding="utf-8")])
    if len(roots) != 1:
        errors.append(f"Expected exactly one main LaTeX document; found {len(roots)}")
        return {"status": "HOLD", "chapter": 9, "source_files": [str(x.relative_to(root)) for x in tex_files],
                "figure_environments": 0, "checked_graphics": 0, "errors": errors}
    try:
        content, parts = read_tex_tree(roots[0], manuscript)
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(str(exc))
        return {"status": "HOLD", "chapter": 9, "source_files": [str(x.relative_to(root)) for x in tex_files],
                "figure_environments": 0, "checked_graphics": 0, "errors": errors}
    sections = figures_in_tex(content)
    all_refs = set(REFERENCE.findall(content))
    graphics = 0
    for i, section in enumerate(sections, 1):
        if not section["has_caption"]:
            errors.append(f"figure environment {i}: missing caption")
        if not section["graphics"]:
            errors.append(f"figure environment {i}: missing includegraphics")
        if len(section["labels"]) != 1:
            errors.append(f"figure environment {i}: expected exactly one label")
        for label in section["labels"]:
            if label not in all_refs:
                errors.append(f"figure environment {i}: orphan label {label} not referenced in prose")
        for name in section["graphics"]:
            graphics += 1
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"figure environment {i}: unsafe graphics path {name}")
                continue
            candidates = [roots[0].parent / path, chapter / path,
                          chapter / "figures" / path]
            ext = [".pdf", ".png", ".svg", ".jpg", ".jpeg"]
            found = any(p.is_file() for candidate in candidates for p in (
                [candidate] if candidate.suffix else [candidate.with_suffix(e) for e in ext]))
            if not found:
                errors.append(f"figure environment {i}: asset missing {name}")
    if len(sections) != 11:
        errors.append(f"Expected 11 final figure environments; found {len(sections)}")
    if not (chapter / "manuscript" / "chapter9.pdf").is_file():
        errors.append("Final compiled chapter9.pdf is missing")
    return {"status": "HOLD" if errors else "STRUCTURALLY_VERIFIED",
            "chapter": 9, "source_files": [str(x.relative_to(root)) for x in parts],
            "figure_environments": len(sections), "checked_graphics": graphics,
            "errors": errors,
            "limitations": "Even STRUCTURALLY_VERIFIED requires scientific/visual and textual expert review."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    a = p.parse_args()
    result = audit(a.root)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Chapter 9 manuscript integration: {result['status']}; {len(result['errors'])} issue(s)")
    for issue in result["errors"]:
        print(" - " + issue)
    return int(a.strict and result["status"] != "STRUCTURALLY_VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
