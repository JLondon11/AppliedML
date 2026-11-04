"""Chapter 5 case-study source-level asset audit (not experimental validation).

Usage: python audit_case_study_assets.py /path/to/Chapter_5.tex
Writes an adjacent case_study_source_asset_audit.csv.
The script checks section-local figure/table presence and referenced asset
existence; a human must inspect model identity, dataset and evidence provenance.
"""
import csv
import re
import sys
from pathlib import Path

tex = Path(sys.argv[1]).resolve()
s = tex.read_text(encoding="utf-8")
heads = list(re.finditer(r"\\(?:section|subsection|subsubsection)\{([^{}]+)\}",s))
rows=[]
for i,h in enumerate(heads):
    title=h.group(1)
    if not re.search(r"Case Study|Application",title,re.I):
        continue
    if title!="YOLO Autonomous Drone Application" and not re.search(
            r"Case Study|Application:",title,re.I):
        continue
    end=heads[i+1].start() if i+1<len(heads) else len(s)
    body=s[h.end():end]
    figs=re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}",body,re.S)
    tabs=re.findall(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}",body,re.S)
    assets=[p for env in figs for p in re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",env)]
    missing=[p for p in assets if not (tex.parent/p).exists() and not
        (tex.parent/"Figures"/Path(p).name).exists()]
    rows.append(dict(section=title,figure_count=len(figs),table_count=len(tabs),
                     figure_assets=";".join(assets),missing_assets=";".join(missing),
                     citation_calls=len(re.findall(r"\\cite(?:\[[^]]*\])?\{",body)),
                     model_inference_verified="NOT ASSESSED"))
out=tex.parent/"case_study_source_asset_audit.csv"
with out.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f"{len(rows)} studies; {sum(r['figure_count'] for r in rows)} figures; "
      f"{sum(r['table_count'] for r in rows)} tables; "
      f"{sum(bool(r['missing_assets']) for r in rows)} studies missing figure files")
print(out)
