#!/usr/bin/env python3
"""Audit final chapter trees for Springer-oriented publication completeness."""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

PROBLEM_PAT = re.compile(r"\\(?:section|chapter|subsection)\*?\{[^}]*?(?:Problems|Exercises|Problem Set)[^}]*\}", re.I)
FIG_LABEL = re.compile(r"\\label\{fig:[^}]+\}")
TAB_LABEL = re.compile(r"\\label\{tab:[^}]+\}")
LST_LABEL = re.compile(r"\\label\{lst:[^}]+\}")
BAD_FONTS = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{(?:fontspec|mathpazo|times|newtxtext|newtxmath|libertine|ebgaramond|fourier|charter)\}", re.I)
FLEQN = re.compile(r"\bfleqn\b", re.I)

def stems(root: Path, ext: str):
    return {p.relative_to(root).with_suffix("") for p in root.rglob(f"*.{ext}")}

def audit_chapter(ch: Path):
    flags=[]
    tex=list(ch.glob("*.tex"))
    pdf=list(ch.glob("*.pdf"))
    if not tex: flags.append(("missing-tex","No final root .tex file at chapter root"))
    if not pdf: flags.append(("missing-pdf","No compiled chapter PDF at chapter root"))
    if tex:
        # Prefer a plausible final root file, otherwise inspect all root tex files together.
        text="\n".join(p.read_text(encoding="utf-8",errors="replace") for p in tex)
        if not PROBLEM_PAT.search(text):
            flags.append(("missing-problem-set","No end-of-chapter Problems/Exercises section detected"))
        if BAD_FONTS.search(text):
            flags.append(("custom-font-package","Custom body-font package detected; Springer book template/default typography should control fonts"))
        if FLEQN.search(text):
            flags.append(("fleqn","fleqn detected; display equations must remain centered"))
        for kind,pat in [("figure",FIG_LABEL),("table",TAB_LABEL),("listing",LST_LABEL)]:
            if not pat.search(text):
                flags.append((f"no-{kind}-labels",f"No chapter-scoped {kind} labels detected in root source"))
    figs=ch/"figures"
    if not figs.exists():
        flags.append(("missing-figures-dir","No figures directory"))
    else:
        sets={e:stems(figs,e) for e in ("png","svg","pdf")}
        union=set().union(*sets.values())
        for s in sorted(union):
            have={e for e,v in sets.items() if s in v}
            if have != {"png","svg","pdf"}:
                flags.append(("figure-triformat",f"{s}: has {sorted(have)}, requires png/svg/pdf"))
    code=ch/"code"
    if not code.exists():
        flags.append(("missing-code-dir","No code directory"))
    if not (code/"LISTING_MANIFEST.md").exists():
        flags.append(("missing-listing-manifest","code/LISTING_MANIFEST.md is missing"))
    alt_candidates=[ch/"FIGURE_ALT_TEXT.md",ch/"figure_alt_text.md",ch/"figures"/"ALT_TEXT.md"]
    if not any(p.exists() for p in alt_candidates):
        flags.append(("missing-alt-text","No chapter-level figure alt-text manifest"))
    return flags

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",nargs="?",default="book/final")
    args=ap.parse_args()
    root=Path(args.root)
    chapters=[p for p in sorted(root.iterdir()) if p.is_dir() and re.match(r"^\d{2}_",p.name)]
    total=0
    for ch in chapters:
        flags=audit_chapter(ch)
        total+=len(flags)
        print(f"[{ch.name}] flags={len(flags)}")
        for k,d in flags: print(f"  {k}: {d}")
    print(f"chapters_scanned={len(chapters)}")
    print(f"springer_completeness_flags={total}")
    return 1 if total else 0

if __name__=="__main__":
    raise SystemExit(main())
