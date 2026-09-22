#!/usr/bin/env python3
"""Audit case-study/application structure in LaTeX manuscripts.

Checks:
- several substantive introductory paragraphs before first structured lead-in;
- bolded lead-in and prose on the same line;
- no isolated lead-in-only lines;
- no repeated generic lead-ins within one case-study/application section;
- no near-duplicate labeled paragraphs by token-set similarity.

This is a conservative production QA tool. Findings require editorial review.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

SECTION_RE = re.compile(r"\\(?:sub)*section\*?\{([^}]*)\}", re.I)
CASE_RE = re.compile(r"\b(case\s*study|application)\b", re.I)

LEADINS = [
    "Problem", "Dataset", "Data", "Method", "Model", "Experimental Setup",
    "Results", "Evaluation", "Limitations", "Deployment", "Lessons Learned",
    "Implications", "Business Impact"
]
LEADIN_ALT = "|".join(re.escape(x) for x in sorted(LEADINS, key=len, reverse=True))
INLINE_RE = re.compile(
    rf"^\s*\\textbf\{{(?P<label>{LEADIN_ALT})\.\}}\s+(?P<text>\S.*)$",
    re.I
)
ISOLATED_RE = re.compile(
    rf"^\s*\\textbf\{{(?P<label>{LEADIN_ALT})\.?\}}\s*$",
    re.I
)
ANY_LEAD_RE = re.compile(
    rf"\\textbf\{{(?P<label>{LEADIN_ALT})\.?\}}",
    re.I
)

CMD_RE = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")

def plain(s: str) -> str:
    s = COMMENT_RE.sub("", s)
    s = CMD_RE.sub(" ", s)
    s = s.replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()

def toks(s: str):
    stop={"the","a","an","and","or","of","to","in","is","are","was","were","for","on","with","that","this","as","by","from","it","be","has","have","had"}
    return {w.lower() for w in WORD_RE.findall(plain(s)) if len(w)>2 and w.lower() not in stop}

def jaccard(a,b):
    if not a or not b: return 0.0
    return len(a & b)/len(a | b)

def split_paragraphs(lines):
    out=[]; cur=[]; start=None
    for ln,text in lines:
        if not text.strip():
            if cur:
                out.append((start," ".join(x for _,x in cur).strip()))
                cur=[]; start=None
            continue
        if start is None: start=ln
        cur.append((ln,text.strip()))
    if cur: out.append((start," ".join(x for _,x in cur).strip()))
    return out

def audit_file(path: Path, min_intro_paragraphs=3, near_dup=0.72):
    raw=path.read_text(encoding="utf-8",errors="replace").splitlines()
    sections=[]
    current=None
    for i,line in enumerate(raw,1):
        m=SECTION_RE.search(COMMENT_RE.sub("",line))
        if m:
            if current: sections.append(current)
            current={"title":m.group(1),"start":i,"lines":[]}
        elif current:
            current["lines"].append((i,line))
    if current: sections.append(current)

    findings=[]
    for sec in sections:
        if not CASE_RE.search(sec["title"]): continue
        lines=[(ln,COMMENT_RE.sub("",txt)) for ln,txt in sec["lines"]]
        pars=split_paragraphs(lines)

        labeled=[]
        first_label_idx=None
        for pi,(ln,p) in enumerate(pars):
            m=ANY_LEAD_RE.match(p)
            if m:
                label=m.group("label").strip().lower()
                labeled.append((label,ln,p))
                if first_label_idx is None: first_label_idx=pi

        if not labeled:
            findings.append(dict(file=str(path),section=sec["title"],line=sec["start"],
                kind="no-structured-leadins",label="",detail="No recognized bolded structured lead-ins found."))
            continue

        intro=pars[:first_label_idx]
        substantive=[p for _,p in intro if len(WORD_RE.findall(plain(p)))>=35]
        if len(substantive)<min_intro_paragraphs:
            findings.append(dict(file=str(path),section=sec["title"],line=labeled[0][1],
                kind="insufficient-introduction",label="",
                detail=f"Only {len(substantive)} substantive introductory paragraphs before first structured lead-in; require at least {min_intro_paragraphs}."))

        # line-level formatting
        for ln,line in lines:
            iso=ISOLATED_RE.match(line)
            if iso:
                findings.append(dict(file=str(path),section=sec["title"],line=ln,
                    kind="isolated-leadin",label=iso.group("label"),
                    detail="Bolded lead-in appears on a line by itself; prose must continue on same line."))
            elif ANY_LEAD_RE.match(line):
                m=INLINE_RE.match(line)
                if not m:
                    findings.append(dict(file=str(path),section=sec["title"],line=ln,
                        kind="lead-in-format",label=ANY_LEAD_RE.match(line).group("label"),
                        detail="Lead-in must be bolded, end with a period inside the bold span, and be followed immediately by prose on the same line."))

        counts=Counter(label for label,_,_ in labeled)
        for label,n in counts.items():
            if n>1:
                findings.append(dict(file=str(path),section=sec["title"],line=sec["start"],
                    kind="duplicate-leadin",label=label,
                    detail=f"Generic lead-in appears {n} times in this case study/application."))

        # near-duplicate semantic-ish token overlap among labeled paragraphs
        for i in range(len(labeled)):
            l1,ln1,p1=labeled[i]; t1=toks(p1)
            for j in range(i+1,len(labeled)):
                l2,ln2,p2=labeled[j]; t2=toks(p2)
                sim=jaccard(t1,t2)
                if sim>=near_dup and len(t1)>=8 and len(t2)>=8:
                    findings.append(dict(file=str(path),section=sec["title"],line=ln2,
                        kind="near-duplicate-labeled-paragraph",label=f"{l1}/{l2}",
                        detail=f"Token-set Jaccard similarity {sim:.2f}; review for redundant content with paragraph at line {ln1}."))
    return findings

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("paths",nargs="*",default=["."])
    ap.add_argument("--csv",dest="csv_path")
    ap.add_argument("--min-intro-paragraphs",type=int,default=3)
    ap.add_argument("--near-duplicate-threshold",type=float,default=0.72)
    args=ap.parse_args()

    files=[]
    for item in args.paths:
        p=Path(item)
        if p.is_file() and p.suffix.lower()==".tex": files.append(p)
        elif p.is_dir(): files.extend(p.rglob("*.tex"))

    findings=[]
    for p in sorted(set(files)):
        findings.extend(audit_file(p,args.min_intro_paragraphs,args.near_duplicate_threshold))

    if args.csv_path:
        out=Path(args.csv_path); out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=["file","section","line","kind","label","detail"])
            w.writeheader(); w.writerows(findings)

    print(f"manuscript_files_scanned={len(set(files))}")
    print(f"case_study_application_flags={len(findings)}")
    for x in findings:
        print(f"{x['file']}:{x['line']}: [{x['section']}] {x['kind']} {x['label']} — {x['detail']}")

    if not files:
        print("ERROR: no .tex manuscript sources found; case-study/application QA cannot certify the manuscript.",file=sys.stderr)
        return 2
    return 1 if findings else 0

if __name__=="__main__":
    raise SystemExit(main())
