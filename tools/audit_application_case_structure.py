#!/usr/bin/env python3
"""Audit Application/Case Study narrative-first structure and inline bold fields."""
from __future__ import annotations
import argparse,re,sys
from pathlib import Path

HEAD=re.compile(r"\\(?:section|subsection|subsubsection)\*?\{([^}]*(?:Application|Case Study)[^}]*)\}",re.I)
FIELD=re.compile(r"\\textbf\{([^}:]+):\}\s+([^\n].*)")
BARE_FIELD=re.compile(r"\\textbf\{([^}:]+)\}\s*$")
COMMENT=re.compile(r"(?<!\\)%.*$")
BEGIN=re.compile(r"\\begin\{([^}]+)\}")
END=re.compile(r"\\end\{([^}]+)\}")
IGNORE={"figure","figure*","table","table*","tabular","tabularx","longtable","lstlisting","equation","equation*","align","align*","gather","gather*"}

def substantive(s,min_words=35):
    s=re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?"," ",s)
    s=re.sub(r"\$[^$]*\$|\\\([^)]*\\\)"," MATH ",s)
    words=re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*",s)
    return len(words)>=min_words

def audit(path:Path):
    lines=path.read_text(encoding="utf-8",errors="replace").splitlines()
    flags=[]; in_target=False; title=""; intro=[]; field_started=False; env=[]
    def finish():
        nonlocal intro,in_target,title,field_started
        if in_target and len(intro)<3:
            flags.append((title,f"only {len(intro)} substantive introductory/background paragraphs before structured fields"))
        intro=[]; in_target=False; title=""; field_started=False
    para=[]
    def flush():
        nonlocal para
        if in_target and not field_started and substantive(" ".join(para)):
            intro.append(" ".join(para))
        para=[]
    for i,line in enumerate(lines,1):
        s=COMMENT.sub("",line)
        hm=HEAD.search(s)
        if hm:
            flush(); finish()
            in_target=True; title=hm.group(1); continue
        if in_target and re.match(r"\\(?:section|subsection|subsubsection)\*?\{",s) and not HEAD.search(s):
            flush(); finish(); continue
        bm=BEGIN.search(s)
        if bm: flush(); env.append(bm.group(1)); continue
        em=END.search(s)
        if em:
            flush()
            if env and env[-1]==em.group(1): env.pop()
            continue
        if env and env[-1] in IGNORE: continue
        if not in_target: continue
        if BARE_FIELD.search(s):
            flags.append((title,f"line {i}: standalone bold field label must be inline with prose"))
            field_started=True; flush(); continue
        fm=FIELD.search(s)
        if fm:
            flush(); field_started=True
            if not substantive(fm.group(2),min_words=8):
                flags.append((title,f"line {i}: structured field '{fm.group(1)}' has insufficient prose"))
            continue
        if not s.strip(): flush(); continue
        para.append(s)
    flush(); finish()
    return flags

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("paths",nargs="+")
    args=ap.parse_args(); files=[]
    for item in args.paths:
        p=Path(item)
        if p.is_file() and p.suffix==".tex": files.append(p)
        elif p.is_dir(): files.extend(p.rglob("*.tex"))
    flags=[]
    for p in sorted(set(files)):
        for title,msg in audit(p): flags.append((p,title,msg))
    print(f"files_scanned={len(set(files))}")
    print(f"application_case_structure_flags={len(flags)}")
    for p,title,msg in flags: print(f"{p}: {title}: {msg}")
    return 1 if flags else 0

if __name__=="__main__":
    raise SystemExit(main())
