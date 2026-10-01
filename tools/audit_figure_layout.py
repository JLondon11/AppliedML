#!/usr/bin/env python3
"""Audit final-book LaTeX figure sizing and vertical-spacing conventions."""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path

FIG_RE=re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}",re.S)
INC_RE=re.compile(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}")
WIDTH_RE=re.compile(r"width\s*=\s*([^,\]]+)")
VSPACE_RE=re.compile(r"\\vspace\*?\{\s*([^}]+)\}")
CANON={"\\BookFigureWidth","\\BookWideFigureWidth","\\BookCompactFigureWidth"}

def positive_vspace(expr:str)->bool:
    s=expr.strip().replace(" ","")
    if s.startswith("-"): return False
    if s in {"0","0pt","0mm","0cm","0em","0ex"}: return False
    return True

def audit(path:Path):
    txt=path.read_text(encoding="utf-8",errors="replace")
    out=[]
    for m in FIG_RE.finditer(txt):
        block=m.group(1)
        line=txt.count("\n",0,m.start())+1
        for vm in VSPACE_RE.finditer(block):
            if positive_vspace(vm.group(1)):
                out.append(dict(file=str(path),line=line,kind="positive-vspace",
                                detail=f"Positive vertical space inside figure: {vm.group(1)}"))
        incs=list(INC_RE.finditer(block))
        if not incs:
            out.append(dict(file=str(path),line=line,kind="figure-without-graphics",
                            detail="Figure environment contains no includegraphics call."))
            continue
        for im in incs:
            opts=im.group(1) or ""
            w=WIDTH_RE.search(opts)
            if not w:
                out.append(dict(file=str(path),line=line,kind="missing-width",
                                detail=f"{im.group(2)} has no explicit canonical width."))
                continue
            val=w.group(1).strip()
            if val not in CANON:
                out.append(dict(file=str(path),line=line,kind="noncanonical-width",
                                detail=f"{im.group(2)} uses width={val}; use a Book*FigureWidth macro."))

    # Also catch positive vspace immediately adjacent to a figure environment.
    lines=txt.splitlines()
    for i,line in enumerate(lines):
        if "\\begin{figure" in line or "\\end{figure" in line:
            for j in range(max(0,i-2),min(len(lines),i+3)):
                vm=VSPACE_RE.search(lines[j])
                if vm and positive_vspace(vm.group(1)):
                    out.append(dict(file=str(path),line=j+1,kind="adjacent-positive-vspace",
                                    detail=f"Positive vertical space adjacent to figure: {vm.group(1)}"))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("paths",nargs="*",default=["book/final"])
    ap.add_argument("--csv")
    args=ap.parse_args()
    files=[]
    for x in args.paths:
        p=Path(x)
        if p.is_file() and p.suffix==".tex": files.append(p)
        elif p.is_dir(): files.extend(p.rglob("*.tex"))
    files=[p for p in files if "book/final/shared" not in p.as_posix()]
    findings=[]
    for p in sorted(set(files)): findings.extend(audit(p))
    if args.csv:
        q=Path(args.csv); q.parent.mkdir(parents=True,exist_ok=True)
        with q.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=["file","line","kind","detail"])
            w.writeheader(); w.writerows(findings)
    print(f"figure_layout_files_scanned={len(set(files))}")
    print(f"figure_layout_flags={len(findings)}")
    for x in findings:
        print(f"{x['file']}:{x['line']}: {x['kind']} — {x['detail']}")
    return 1 if findings else 0

if __name__=="__main__":
    raise SystemExit(main())
