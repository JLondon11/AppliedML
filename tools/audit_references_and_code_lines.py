#!/usr/bin/env python3
"""Audit figure/table/listing references and code-line numbering in LaTeX sources."""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path

LABEL_RE=re.compile(r"\\label\{((?:fig|tab|lst):[^}]+)\}")
REF_RE=re.compile(r"\\(?:ref|autoref|cref|Cref)\{([^}]+)\}")
LISTING_BEGIN=re.compile(r"\\begin\{lstlisting\}(?:\[([^\]]*)\])?")
CAPTION_OPT=re.compile(r"(?:^|,)\s*caption\s*=\s*\{?([^,}]+)")
LABEL_OPT=re.compile(r"(?:^|,)\s*label\s*=\s*\{?([^,}]+)")
STYLE_OPT=re.compile(r"(?:^|,)\s*style\s*=\s*BookCode\b")

def audit(path:Path):
    text=path.read_text(encoding="utf-8",errors="replace")
    findings=[]
    labels=[(m.group(1), text.count("\n",0,m.start())+1) for m in LABEL_RE.finditer(text)]
    refs=set()
    for m in REF_RE.finditer(text):
        for ref in m.group(1).split(","):
            refs.add(ref.strip())
    for label,line in labels:
        if label not in refs:
            findings.append(dict(file=str(path),line=line,kind="orphan-label",label=label,
                detail="Figure/table/listing label is not referenced by prose with ref/autoref/cref."))

    for m in LISTING_BEGIN.finditer(text):
        line=text.count("\n",0,m.start())+1
        opts=m.group(1) or ""
        if not CAPTION_OPT.search(opts):
            findings.append(dict(file=str(path),line=line,kind="listing-missing-caption",label="",
                detail="Listing must have a descriptive title/caption."))
        lab=LABEL_OPT.search(opts)
        if not lab:
            findings.append(dict(file=str(path),line=line,kind="listing-missing-label",label="",
                detail="Listing must have a unique lst: label."))
        if not STYLE_OPT.search(opts):
            findings.append(dict(file=str(path),line=line,kind="listing-style-not-enforced",label=lab.group(1) if lab else "",
                detail="Listing must use style=BookCode so every code line is numbered."))

    # Ban explicit disabling of line numbers.
    for i,line in enumerate(text.splitlines(),1):
        if re.search(r"\bnumbers\s*=\s*none\b",line):
            findings.append(dict(file=str(path),line=i,kind="line-numbering-disabled",label="",
                detail="All code lines must be numbered; numbers=none is prohibited."))
    return findings

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("paths",nargs="*",default=["book/final"])
    ap.add_argument("--csv")
    args=ap.parse_args()
    files=[]
    for s in args.paths:
        p=Path(s)
        if p.is_file() and p.suffix==".tex": files.append(p)
        elif p.is_dir(): files.extend(p.rglob("*.tex"))
    findings=[]
    for p in sorted(set(files)): findings.extend(audit(p))
    if args.csv:
        out=Path(args.csv); out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=["file","line","kind","label","detail"])
            w.writeheader(); w.writerows(findings)
    print(f"files_scanned={len(set(files))}")
    print(f"reference_listing_flags={len(findings)}")
    for x in findings:
        print(f"{x['file']}:{x['line']}: {x['kind']} {x['label']} — {x['detail']}")
    return 1 if findings else 0

if __name__=="__main__":
    raise SystemExit(main())
