"""Strict print-width and PDF-text bounds audit of seven Chapter 6 plates.

Does not claim final approval without the actual latest chapter PDF/TeX.
"""
from pathlib import Path
import argparse,hashlib,json
import fitz
NAMES=[
"tissue_source_compact.pdf","digits_source_compact.pdf","nci_source_compact.pdf",
"tissue_vae_output_compact_1.pdf","tissue_vae_output_compact_2.pdf",
"smiles_vae_architecture_compact.pdf","tissue_vae_architecture_compact.pdf"]
EXPECTED_IMAGES={
"tissue_source_compact.pdf":7,"digits_source_compact.pdf":20,
"nci_source_compact.pdf":6,"tissue_vae_output_compact_1.pdf":16,
"tissue_vae_output_compact_2.pdf":16,
"smiles_vae_architecture_compact.pdf":0,
"tissue_vae_architecture_compact.pdf":0}
def audit(directory):
    out=[];issues=[]
    for name in NAMES:
        path=Path(directory)/name
        if not path.exists():issues.append("Missing "+name);continue
        with fitz.open(path) as doc:
            if len(doc)!=1:issues.append("Not single-page: "+name);continue
            page=doc[0];spans=[]
            for b in page.get_text("dict")["blocks"]:
                for line in b.get("lines",[]):
                    spans.extend(s for s in line.get("spans",[]) if s["text"].strip())
            images=len(page.get_images(full=True))
            if images!=EXPECTED_IMAGES[name]:
                issues.append(f"{name}: expected {EXPECTED_IMAGES[name]} real or model image panels; got {images}")
            if not spans:issues.append(name+": no true selectable vector text");continue
            size=min(s["size"] for s in spans)
            scale=min(1,468/page.rect.width)
            physically_printed=size*scale
            if abs(page.rect.width-468)>.1:issues.append(name+": not the requested 468pt physical figure width")
            if physically_printed<7.5:issues.append(f"{name}: physically printed text too small {physically_printed:.2f}pt")
            clipped=[]
            for s in spans:
                x0,y0,x1,y1=s["bbox"]
                if x0<-.1 or y0<-.1 or x1>page.rect.width+.1 or y1>page.rect.height+.1:
                    clipped.append(s["text"][:60])
            if clipped:issues.append(name+": off-page text: "+"; ".join(clipped))
            out.append(dict(figure=name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                page_width_pt=round(page.rect.width,2),embedded_images=images,
                vector_spans=len(spans),minimum_physically_printed_font_pt=round(physically_printed,2),
                offpage_text=clipped,print_preflight_pass=physically_printed>=7.5 and not clipped))
    return {"gate":"PASS_STAGED_ASSETS_ONLY" if not issues else "HOLD",
      "all_expected_figures":7,"figures_inspected":len(out),"pass_count":sum(x["print_preflight_pass"] for x in out),
      "warnings":issues,"figures":out,"true_chapter_manuscript_verified":False,
      "production_approved":False,
      "limitation":"Assumed 6.5-inch text width; not based on access to current complete Chapter 6 source."}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--report",type=Path)
    a=p.parse_args()
    result=audit(a.directory);body=json.dumps(result,indent=2)+"\n";print(body,end="")
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(body)
    if result["warnings"]:raise SystemExit(1)
