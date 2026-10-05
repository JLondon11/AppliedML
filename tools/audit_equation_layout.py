#!/usr/bin/env python3
"""Audit LaTeX sources for violations of the centered-display-equation rule."""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

FLEQN_DOC = re.compile(r"\\documentclass\s*\[([^\]]*\bfleqn\b[^\]]*)\]", re.I)
FLEQN_AMS = re.compile(r"\\usepackage\s*\[([^\]]*\bfleqn\b[^\]]*)\]\s*\{amsmath\}", re.I)
MATHINDENT = re.compile(r"\\(?:setlength|addtolength)\s*\{\\mathindent\}", re.I)
DISPLAY_START = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}|\\\[")
SHIFT = re.compile(r"\\(?:hspace\*?|kern)\s*\{?\s*-?\d")
COMMENT = re.compile(r"(?<!\\)%.*$")

def audit(path: Path):
    raw=path.read_text(encoding="utf-8",errors="replace").splitlines()
    findings=[]
    in_display=False; env=None
    for n,line in enumerate(raw,1):
        s=COMMENT.sub("",line)
        if FLEQN_DOC.search(s):
            findings.append((n,"fleqn-documentclass","documentclass uses fleqn; displayed equations must be centered"))
        if FLEQN_AMS.search(s):
            findings.append((n,"fleqn-amsmath","amsmath is loaded with fleqn"))
        if MATHINDENT.search(s):
            findings.append((n,"mathindent","manual mathindent modification is prohibited"))
        m=DISPLAY_START.search(s)
        if m:
            in_display=True
            env=m.group(1) if m.groups() else "display"
        if in_display and SHIFT.search(s):
            findings.append((n,"manual-display-shift","manual horizontal shifting inside display math requires review"))
        if in_display and ("\\end{"+str(env)+"}" in s if env and env!="display" else "\\]" in s):
            in_display=False; env=None
    return findings

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("paths",nargs="*",default=["book/final"])
    args=ap.parse_args()
    files=[]
    for item in args.paths:
        p=Path(item)
        if p.is_file() and p.suffix==".tex": files.append(p)
        elif p.is_dir(): files.extend(p.rglob("*.tex"))
    findings=[]
    for p in sorted(set(files)):
        for n,k,d in audit(p): findings.append((p,n,k,d))
    print(f"files_scanned={len(set(files))}")
    print(f"equation_layout_flags={len(findings)}")
    for p,n,k,d in findings: print(f"{p}:{n}: {k} - {d}")
    return 1 if findings else 0

if __name__=="__main__":
    raise SystemExit(main())
