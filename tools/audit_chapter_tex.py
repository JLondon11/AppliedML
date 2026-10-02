#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

LABEL_RE=re.compile(r"\\label\{([^}]+)\}")
REF_RE=re.compile(r"\\(?:ref|autoref|eqref)\{([^}]+)\}")
INPUT_RE=re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHICS_RE=re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
FIG_ENV_RE=re.compile(r"\\begin\{figure\}.*?\\end\{figure\}",re.S)
TAB_ENV_RE=re.compile(r"\\begin\{table\}.*?\\end\{table\}",re.S)
LST_ENV_RE=re.compile(r"\\begin\{lstlisting\}(?:\[([^\]]*)\])?.*?\\end\{lstlisting\}",re.S)

def read_recursive(path:Path, seen:set[Path]) -> str:
    path=path.resolve()
    if path in seen: return ""
    seen.add(path)
    text=path.read_text(encoding="utf-8")
    base=path.parent
    parts=[text]
    for name in INPUT_RE.findall(text):
        p=(base/name)
        if not p.suffix: p=p.with_suffix(".tex")
        if p.exists(): parts.append(read_recursive(p,seen))
    return "\n".join(parts)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--chapter-dir",default=None)
    args=ap.parse_args()
    tex=Path(args.tex).resolve()
    root=Path(args.chapter_dir).resolve() if args.chapter_dir else tex.parent
    seen=set()
    text=read_recursive(tex,seen)
    errors=[]; warnings=[]

    labels=LABEL_RE.findall(text)
    refs=REF_RE.findall(text)
    dup=sorted({x for x in labels if labels.count(x)>1})
    missing=sorted(set(refs)-set(labels))
    if dup: errors.append({"duplicate_labels":dup})
    if missing: errors.append({"unresolved_internal_refs":missing})

    # Every figure/table/listing should have a label and be cited outside itself.
    for kind,rx in [("figure",FIG_ENV_RE),("table",TAB_ENV_RE)]:
        for i,block in enumerate(rx.findall(text),1):
            ls=LABEL_RE.findall(block)
            if not ls:
                errors.append({f"{kind}_without_label":i}); continue
            for lab in ls:
                outside=text.replace(block,"",1)
                if not re.search(r"\\(?:ref|autoref)\{"+re.escape(lab)+r"\}",outside):
                    errors.append({f"unreferenced_{kind}":lab})

    for i,m in enumerate(LST_ENV_RE.finditer(text),1):
        opts=m.group(1) or ""
        block=m.group(0)
        ls=LABEL_RE.findall(block)
        # lstlisting labels are commonly specified in options.
        optlab=re.search(r"(?:^|,)\s*label\s*=\s*\{?([^,}\]]+)\}?",opts)
        if optlab: ls.append(optlab.group(1).strip())
        cap=re.search(r"(?:^|,)\s*caption\s*=\s*\{?([^,\]]+)",opts)
        if not cap: errors.append({"listing_without_caption":i})
        if not ls: errors.append({"listing_without_label":i})
        for lab in ls:
            outside=text[:m.start()]+text[m.end():]
            if not re.search(r"\\(?:ref|autoref)\{"+re.escape(lab)+r"\}",outside):
                errors.append({"unreferenced_listing":lab})

    # Resolve graphics using TeX-like extension fallback.
    for g in GRAPHICS_RE.findall(text):
        gp=Path(g)
        candidates=[]
        bases=[tex.parent/gp,root/gp]
        for b in bases:
            if b.suffix: candidates.append(b)
            else: candidates.extend(b.with_suffix(e) for e in (".pdf",".png",".jpg",".jpeg"))
        if not any(c.exists() for c in candidates):
            errors.append({"missing_graphic":g})

    # Hard editorial contamination patterns.
    bad_patterns=[
      r"final figure should",r"captionless placeholder",r"original chapter",
      r"original code",r"this replaces",r"Application Framework\.",r"Case Study Framework\."
    ]
    for pat in bad_patterns:
        n=len(re.findall(pat,text,re.I))
        if n: errors.append({"editorial_contamination":pat,"count":n})

    report={
      "root_tex":str(tex),
      "tex_files_read":[str(p) for p in sorted(seen)],
      "labels":len(labels),"references":len(refs),
      "figures":len(FIG_ENV_RE.findall(text)),
      "tables":len(TAB_ENV_RE.findall(text)),
      "listings":len(list(LST_ENV_RE.finditer(text))),
      "errors":errors,"warnings":warnings,
      "status":"PASS" if not errors else "FAIL"
    }
    print(json.dumps(report,indent=2))
    sys.exit(0 if not errors else 1)

if __name__=="__main__":
    main()
