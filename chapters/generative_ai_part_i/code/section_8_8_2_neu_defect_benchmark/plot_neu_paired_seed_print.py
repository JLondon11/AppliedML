"""Native 468pt original-data NEU paired seed-effect scientific SVG.

Execute with the retained authentic 12-row NEU source; no GAN retraining
or new imagined observations. Designed at 6.5in assumed final text width
with a minimum vector font size of 10.5pt (>7.5pt legibility floor).
"""
import csv
import hashlib
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parent
CSV=ROOT/"replicated_seed_metrics_pass13.csv"
SOURCE_SHA="1f74ab0c8067612a072c2eff5554b971682ce851bd4c7ca5eb585fbb5f163d9e"
SEEDS=(20261106,20261107,20261108)
ARMS=("real_only","flip_augmentation","gan_augmented","real_repeat_augmented")

def render() -> str:
    if hashlib.sha256(CSV.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("Authentic measured seed results changed")
    with CSV.open(newline="",encoding="utf-8") as fh:
        rows=list(csv.DictReader(fh))
    if len(rows)!=12 or {(int(r["seed"]),r["arm"]) for r in rows}!={(s,a) for s in SEEDS for a in ARMS}:
        raise ValueError("Original NEU seed/arm evidence incomplete")
    source={(int(x["seed"]),x["arm"]):x for x in rows}

    def t(x,y,value,pt=12,color="#284858",weight=400,anchor="start"):
        return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial,Helvetica,sans-serif" '
                f'font-size="{pt}" font-weight="{weight}" fill="{color}">{escape(str(value))}</text>')
    def line(x1,y1,x2,y2,col="#dde5e9",w=1):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}"/>'
    def X(d):
        return round(169+(d+45)/45*(444-169),2)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="468" height="555" '
           'viewBox="0 0 468 555" role="img" aria-label="Real NEU defect measured GAN '
           'versus original controls over three training seeds">',
           '<rect width="468" height="555" fill="white"/>',
           t(12,30,"NEU steel defects: measured GAN effects",17,"#193d50",700),
           t(12,52,"360 original held-out photos reused across 3 training seeds",11.5,"#526c7c"),
           t(12,70,"GAN minus matched real-only / repeated-image controls",11.5,"#526c7c")]
    rows_to_draw=[("test_accuracy","real_only","Accuracy","vs original only"),
                  ("test_macro_f1","real_only","Macro-F1","vs original only"),
                  ("test_accuracy","real_repeat_augmented","Accuracy","vs repeats + flips"),
                  ("test_macro_f1","real_repeat_augmented","Macro-F1","vs repeats + flips")]
    for value in (-40,-30,-20,-10,0):
        xx=X(value)
        parts.extend([line(xx,94,xx,450,"#e6edf0"),
                      t(xx,478,str(value),10.5,"#617787",anchor="middle")])
    palette=("#24718a","#b56850","#658a71")
    for i,(metric,baseline,title,caption) in enumerate(rows_to_draw):
        cy=130+i*92
        parts.extend([t(12,cy-8,title,12.8,"#294a5d",700),
                      t(12,cy+10,caption,11,"#566e7b")])
        diffs=[100*(float(source[s,"gan_augmented"][metric])-
                    float(source[s,baseline][metric])) for s in SEEDS]
        if not all(-45<d<0 for d in diffs):
            raise ValueError("Original scientific result moved outside observed negative figure")
        ys=[cy-14,cy,cy+14]
        parts.append(line(X(min(diffs)),cy,X(max(diffs)),cy,"#8a9fa9",2))
        for j,(delta,y) in enumerate(zip(diffs,ys)):
            parts.append(f'<circle cx="{X(delta)}" cy="{y}" r="4.8" fill="{palette[j]}" '
                         'stroke="white" stroke-width="1"/>')
        avg=statistics.mean(diffs)
        parts.extend([line(X(avg),cy-21,X(avg),cy+21,"#193e50",2),
                      t(169,cy+33,f"Mean {avg:+.2f} pp",11.4,"#304e60",700)])
    parts.extend([t(306,497,"percentage points",10.5,"#536d7f",anchor="middle"),
                  line(12,512,456,512,"#e1e7eb"),
                  t(12,532,"Three negative paired effects: exact one-sided sign p = 0.125.",10.5,"#485f6c"),
                  t(12,548,"Only original NEU test photos; no cross-factory validation.",10.5,"#627785"),
                  "</svg>"])
    return "\n".join(parts)+"\n"

if __name__=="__main__":
    out=ROOT/"paired_seed_science"/"neu_paired_seed_print_468pt.svg"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(render(),encoding="utf-8")
    print(out)
