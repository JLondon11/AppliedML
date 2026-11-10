#!/usr/bin/env python3
"""Audit figures under named case-study/application headings in a LaTeX manuscript.

This scanner does NOT infer that experiments were executed. A caption indicating
a measurable result only identifies a candidate for independent certification.
Usage: python tools/audit_manuscript_application_figure_coverage.py chapter.tex
Option: --archive bookchapter.zip --member path/inside.zip
"""
import argparse,json,re,zipfile
from pathlib import Path

HEAD=re.compile(r'\\(section|subsection|subsubsection)\*?\{([^}]*)\}')
FIG=re.compile(r'\\begin\{figure\*?\}([\s\S]*?)\\end\{figure\*?\}')
LEVEL={'section':1,'subsection':2,'subsubsection':3}
NAMED=re.compile(r'^(?:Application:|Case Study:|Sepsis Prediction: A Flagship Clinical AI Case Study)',re.I)
RESULT=re.compile(r'measured model comparison|held.out|test accuracy|actual predictions|validation performance',re.I)
SCHEMA=re.compile(r'architecture|workflow|pipeline|decision loop|application.level view|integration architecture|system architecture|two.stage',re.I)

def audit(text):
    headers=list(HEAD.finditer(text))
    records=[]
    for i,h in enumerate(headers):
        name=h.group(2)
        if not NAMED.search(name):
            continue
        stop=next((n.start() for n in headers[i+1:]
                   if LEVEL[n.group(1)]<=LEVEL[h.group(1)]),len(text))
        figs=[]
        for match in FIG.finditer(text[h.end():stop]):
            body=match.group(1)
            img=re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}',body)
            cap=re.search(r'\\caption\{',body)
            snippet=re.sub(r'\s+',' ',body[cap.end():cap.end()+230]) if cap else ''
            kind=('measured_result_candidate' if RESULT.search(snippet)
                  else 'schematic_candidate' if SCHEMA.search(snippet)
                  else 'manual_review')
            figs.append({'asset':img.group(1) if img else None,
                         'caption_start':snippet[:200],'class':kind})
        records.append({'name':name,'figures':figs,
                        'contains_measured_candidate':any(f['class']=='measured_result_candidate'
                                                          for f in figs),
                        'scientifically_certified':False})
    return {'named_units':len(records),
            'units_with_figures':sum(bool(r['figures']) for r in records),
            'units_with_measured_result_candidates':sum(r['contains_measured_candidate'] for r in records),
            'units_needing_measured_figures_or_manual_review':sum(not r['contains_measured_candidate'] for r in records),
            'caveat':'Lexical caption/source inventory. Manual figure, real-data and measured numerical QA remain mandatory.',
            'units':records}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('tex',nargs='?')
    p.add_argument('--archive')
    p.add_argument('--member')
    args=p.parse_args()
    if args.archive:
        if not args.member:p.error('--member required with --archive')
        with zipfile.ZipFile(args.archive) as z:
            tex=z.read(args.member).decode('utf-8')
    elif args.tex:
        tex=Path(args.tex).read_text(encoding='utf-8')
    else:
        p.error('provide tex path or --archive and --member')
    print(json.dumps(audit(tex),indent=2))

if __name__=='__main__':
    main()
