#!/usr/bin/env python3
"""Audit LaTeX/Markdown manuscript prose for likely incomplete sentences.

This is a conservative detector. It flags suspicious endings for manual review;
it does not auto-rewrite prose.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

DANGLING = {
    "and","or","but","nor","yet","so",
    "because","although","though","while","whereas","if","unless","since",
    "which","that","who","whose","whom",
    "with","without","of","to","for","from","by","into","onto","through",
}

SKIP_ENVS = {
    "equation","equation*","align","align*","gather","gather*","multline","multline*",
    "lstlisting","minted","verbatim","Verbatim","tabular","tabular*","array",
}

BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
END_RE = re.compile(r"\\end\{([^}]+)\}")
CMD_ONLY_RE = re.compile(r"^\s*\\(?:chapter|section|subsection|subsubsection|paragraph|caption|label)\b")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")
SENTENCE_RE = re.compile(r"(.+?[.!?])(?:\s+|$)", re.S)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LATEX_CMD_RE = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
CITE_RE = re.compile(r"\\(?:cite|citep|citet|ref|eqref|autoref|cref|Cref)\*?(?:\[[^\]]*\])?\{[^}]*\}")

def strip_commands(text: str) -> str:
    text = CITE_RE.sub("", text)
    text = LATEX_CMD_RE.sub(" ", text)
    text = text.replace("~", " ")
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\\([^)]*\\\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def iter_prose_lines(path: Path):
    active_skip = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = COMMENT_RE.sub("", raw)
        for env in BEGIN_RE.findall(line):
            if env in SKIP_ENVS:
                active_skip.append(env)
        if active_skip:
            for env in END_RE.findall(line):
                if active_skip and env == active_skip[-1]:
                    active_skip.pop()
            continue
        if not line.strip() or CMD_ONLY_RE.match(line):
            continue
        if line.lstrip().startswith(("-", "*", "|", "#")) and path.suffix.lower() == ".md":
            continue
        yield lineno, line

def audit_file(path: Path):
    findings = []
    paragraph = []
    start_line = None

    def flush():
        nonlocal paragraph, start_line
        if not paragraph:
            return
        raw = " ".join(paragraph)
        prose = strip_commands(raw)
        for sentence in SENTENCE_RE.findall(prose):
            tokens = TOKEN_RE.findall(sentence)
            if not tokens:
                continue
            last = tokens[-1].lower()
            if last in DANGLING:
                findings.append({
                    "file": str(path),
                    "line": start_line or 1,
                    "kind": "dangling-ending",
                    "token": last,
                    "text": sentence.strip()[:300],
                })
        # Paragraphs with prose but no sentence-final punctuation are suspicious.
        if prose and not re.search(r"[.!?]['\"”’)]?\s*$", prose):
            tokens = TOKEN_RE.findall(prose)
            if len(tokens) >= 4:
                findings.append({
                    "file": str(path),
                    "line": start_line or 1,
                    "kind": "unterminated-paragraph",
                    "token": tokens[-1].lower() if tokens else "",
                    "text": prose[-300:],
                })
        paragraph = []
        start_line = None

    prev_line = None
    for lineno, line in iter_prose_lines(path):
        if start_line is None:
            start_line = lineno
        if prev_line is not None and lineno > prev_line + 1:
            flush()
            start_line = lineno
        paragraph.append(line.strip())
        prev_line = lineno
    flush()
    return findings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["."], help="files/directories to audit")
    ap.add_argument("--csv", dest="csv_path", default=None)
    args = ap.parse_args()

    files = []
    for item in args.paths:
        p = Path(item)
        if p.is_file() and p.suffix.lower() in {".tex", ".md"}:
            files.append(p)
        elif p.is_dir():
            files.extend(x for x in p.rglob("*") if x.suffix.lower() in {".tex", ".md"})

    # Exclude repository/QA metadata; this gate targets reader-facing manuscript prose.
    files = [
        p for p in files
        if "book/qa" not in p.as_posix()
        and not p.name.endswith("_POLICY.md")
        and "RELEASE_MANIFEST" not in p.name
        and p.suffix.lower() == ".tex"
    ]

    findings = []
    for p in sorted(set(files)):
        findings.extend(audit_file(p))

    if args.csv_path:
        out = Path(args.csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file","line","kind","token","text"])
            w.writeheader()
            w.writerows(findings)

    print(f"manuscript_files_scanned={len(set(files))}")
    print(f"sentence_completeness_flags={len(findings)}")
    for x in findings:
        print(f"{x['file']}:{x['line']}: {x['kind']} [{x['token']}] {x['text']}")

    # No manuscript sources is a failing gate: the audit cannot certify completeness.
    if not files:
        print("ERROR: no .tex manuscript sources found; sentence-completeness QA cannot certify the book.", file=sys.stderr)
        return 2
    return 1 if findings else 0

if __name__ == "__main__":
    raise SystemExit(main())
