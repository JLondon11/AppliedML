#!/usr/bin/env python3
"""Audit LaTeX section/subsection depth by substantive prose paragraph count."""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

HEADING=re.compile(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}")
BEGIN=re.compile(r"\\begin\{([^}]+)\}")
END=re.compile(r"\\end\{([^}]+)\}")
COMMENT=re.compile(r"(?<!\\)%.*$")
IGNORE_ENVS={
    "figure","figure*","table","table*","tabular","tabularx","longtable",
    "lstlisting","verbatim","equation","equation*","align","align*","gather",
    "gather*","multline","multline*","itemize","enumerate","description"
}
LATEX_ONLY=re.compile(r"^\s*(?:\\(?:label|caption|centering|includegraphics|vspace|hspace|small|footnotesize|scriptsize|normalsize)\b.*)?\s*$")
CMD=re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
MATH_INLINE=re.compile(r"\$[^$]*\$|\\\([^)]*\\\)")
WS=re.compile(r"\s+")

def clean_text(s:str)->str:
    s=COMMENT.sub("",s)
    s=MATH_INLINE.sub(" MATH ",s)
    s=CMD.sub(" ",s)
    s=s.replace("\\"," ")
    s=re.sub(r"[{}]", " ", s)
    return WS.sub(" ",s).strip()

def paragraph_is_substantive(lines:list[str], min_words:int=35)->bool:
    text=" ".join(lines)
    text=clean_text(text)
    if not text:
        return False
    words=re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*",text)
    return len(words)>=min_words

def parse(path:Path,min_words:int):
    lines=path.read_text(encoding="utf-8",errors="replace").splitlines()
    sections=[]
    current=None
    env_stack=[]
    para=[]
    def flush():
        nonlocal para
        if current is not None and paragraph_is_substantive(para,min_words):
            current["paragraphs"]+=1
        para=[]
    for lineno,line in enumerate(lines,1):
        stripped=COMMENT.sub("",line)
        hm=HEADING.search(stripped)
        if hm and not env_stack:
            flush()
            if current is not None:
                sections.append(current)
            current={"level":hm.group(1),"title":hm.group(2).strip(),"line":lineno,"paragraphs":0}
            continue
        bm=BEGIN.search(stripped)
        if bm:
            flush()
            env_stack.append(bm.group(1))
            continue
        em=END.search(stripped)
        if em:
            flush()
            if env_stack:
                if env_stack[-1]==em.group(1):
                    env_stack.pop()
                elif em.group(1) in env_stack:
                    env_stack.remove(em.group(1))
            continue
        if env_stack and env_stack[-1] in IGNORE_ENVS:
            continue
        if current is None:
            continue
        if not stripped.strip():
            flush()
            continue
        if LATEX_ONLY.match(stripped):
            continue
        para.append(stripped)
    flush()
    if current is not None:
        sections.append(current)
    return sections

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("paths",nargs="+")
    ap.add_argument("--minimum",type=int,default=3)
    ap.add_argument("--min-words",type=int,default=35)
    args=ap.parse_args()
    files=[]
    for item in args.paths:
        p=Path(item)
        if p.is_file() and p.suffix==".tex":
            files.append(p)
        elif p.is_dir():
            files.extend(p.rglob("*.tex"))
    flags=[]
    for p in sorted(set(files)):
        for s in parse(p,args.min_words):
            if s["paragraphs"]<args.minimum:
                flags.append((p,s))
    print(f"files_scanned={len(set(files))}")
    print(f"thin_sections={len(flags)}")
    for p,s in flags:
        print(f"{p}:{s['line']}: {s['level']} {{{s['title']}}} -> {s['paragraphs']} substantive paragraphs; minimum={args.minimum}")
    return 1 if flags else 0

if __name__=="__main__":
    raise SystemExit(main())
