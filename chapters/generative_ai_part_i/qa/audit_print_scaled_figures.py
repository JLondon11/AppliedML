"""Figure-font QA at the intended physical Chapter 6 insertion width.

Native PDF font size is NOT the printed font size after shrinking a 15-inch
landscape figure into a normal-width textbook page. Fails normal-width
candidates with illegibly reduced annotations; does not inspect missing TeX.
"""
import argparse,json
from pathlib import Path
import fitz

def audit(figures,target_width_pt=468,min_text_pt=7.5):
    records=[]
    for file in figures:
        with fitz.open(file) as document:
            for page_index,page in enumerate(document):
                spans=[span for block in page.get_text("dict")["blocks"]
                       for line in block.get("lines",[])
                       for span in line.get("spans",[])
                       if span["text"].strip()]
                if not spans:raise ValueError("PDF has no printable text: "+str(file))
                scale=min(1.0,target_width_pt/page.rect.width)
                minimum=min(float(s["size"]) for s in spans)*scale
                clipped=[]
                for span in spans:
                    b=fitz.Rect(span["bbox"])
                    if b.x0<-.1 or b.y0<-.1 or b.x1>page.rect.width+.1 or b.y1>page.rect.height+.1:
                        clipped.append(span["text"][:45])
                records.append({
                    "source":Path(file).name,"page":page_index+1,
                    "native_width_pt":round(page.rect.width,1),
                    "assumed_text_width_pt":target_width_pt,
                    "resizing_scale":round(scale,5),
                    "minimum_printed_font_pt":round(minimum,2),
                    "text_outside_page":clipped,
                    "pass_normal_width":minimum>=min_text_pt and not clipped
                })
    hold=sum(not r["pass_normal_width"] for r in records)
    return {"status":"STAGED_PRINT_PREFLIGHT_HOLD" if hold else "STAGED_PRINT_PREFLIGHT_PASS_MANUSCRIPT_PENDING",
        "assumed_text_width_inches":target_width_pt/72,
        "minimum_printed_font_threshold_pt":min_text_pt,
        "inspected_pages":len(records),"pass_count":len(records)-hold,"hold_count":hold,
        "results":records,
        "production_approved":False,
        "caveat":"A 6.5-inch text width is an explicit planning assumption, NOT a verified latest Chapter 6 manuscript dimension."}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("pdf",nargs="+",type=Path)
    p.add_argument("--width-pt",type=float,default=468)
    p.add_argument("--min-font-pt",type=float,default=7.5)
    p.add_argument("--report",type=Path)
    a=p.parse_args()
    result=audit(a.pdf,a.width_pt,a.min_font_pt)
    content=json.dumps(result,indent=2)+"\n"
    print(content,end="")
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(content)
