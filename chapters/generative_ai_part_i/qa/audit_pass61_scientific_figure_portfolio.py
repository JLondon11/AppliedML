"""Fail-closed preflight of five delivered REAL-source and scientific vector figures.

Input: extracted Chapter 6 Pass60 ZIP plus Pass61 source-architecture PDF.
This checks file hashes/pagination/text bounds and genuine provenance. It does
NOT claim insertion into the unavailable current Chapter 6 manuscript.
"""
import argparse,hashlib,json
from pathlib import Path
try:
 import fitz
except ImportError as exc:
 raise SystemExit("Install pymupdf for PDF page and text-bound inspection") from exc

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCE={
 "original_tissue_source_and_patches.pdf":
   ("original-data-gallery","3211bfab2579ed7b246677f93b88130f8ffbb8e71eb26adc74c3379eb530f416",13),
 "original_scanned_digits_gallery.pdf":
   ("original-data-gallery","e74cfde0be705c00fda8c1c48fa441cd2e07abb112fedff4335c36ff05785264",20),
 "original_nci_chemical_structures.pdf":
   ("original-molecular-structure-gallery","99c51fef486ec4f2dd91caa25017863bd9ca1e92c24c9d1ef30e191f9c39ad04",12),
 "model_verified_gallery/actual_tissue_cvae_reconstruction_gallery.pdf":
   ("actual-executed-model-output-gallery","e1845abf02988cb5d2dbc4c9a52d764ab3d740bb2457c9e56e2466b1c38b6c36",32),
 "nci_molecular_vae_code_architecture.pdf":
   ("source-code-faithful-conceptual-model-architecture",None,0),
 "tissue_convolutional_vae_code_architecture.pdf":
   ("source-code-faithful-conceptual-model-architecture",None,0),
}
def check(stage):
 stage=Path(stage)
 issues=[];records=[]
 for rel,(kind,sha,minimages) in SOURCE.items():
  f=stage/rel
  if not f.is_file() or f.stat().st_size<10000:
   issues.append("Missing original scientific PDF: "+rel)
   continue
  actual=digest(f)
  if sha and sha!=actual:
   issues.append("Original Pass60 PDF SHA256 changed: "+rel)
  with fitz.open(f) as doc:
   if len(doc)!=1:issues.append("Expected one print-layout page: "+rel)
   p=doc[0];bbox=p.rect;images=len(p.get_images(full=True));sizes=[]
   if images<minimages:issues.append("Insufficient real-source or model-output images: "+rel)
   for b in p.get_text("dict")["blocks"]:
    for line in b.get("lines",[]):
     for span in line.get("spans",[]):
      if not span["text"].strip():continue
      sizes.append(span["size"])
      r=fitz.Rect(span["bbox"])
      if r.x0<-.1 or r.y0<-.1 or r.x1>bbox.width+.1 or r.y1>bbox.height+.1:
       issues.append("Clipped/off-page text: "+rel+" "+span["text"][:45])
   if not sizes:
    issues.append("No selectable/editable text in vector PDF: "+rel)
   if sizes and min(sizes)<6:
    issues.append("Very small text (<6 pt): "+rel)
   if rel=="nci_molecular_vae_code_architecture.pdf":
    text=p.get_text()
    for fragment in ("BiGRU encoder","Autoregressive GRU decoder","32 features","120 original","RDKit chemistry"):
     if fragment not in text:issues.append("Architecture label missing: "+fragment)
   if rel=="tissue_convolutional_vae_code_architecture.pdf":
    text=p.get_text()
    for fragment in ("Conv encoder","Transposed-convolution decoder","24 components","589 source-region"):
     if fragment not in text:issues.append("Tissue architecture label missing: "+fragment)
   records.append({"path":rel,"scientific_figure_type":kind,"sha256":actual,
     "images":images,"min_text_pt":round(min(sizes),2) if sizes else None,
     "max_text_pt":round(max(sizes),2) if sizes else None,
     "page_size_pt":[round(bbox.width,1),round(bbox.height,1)]})
 categories={r["scientific_figure_type"] for r in records}
 if sum("architecture" in r["scientific_figure_type"] for r in records)<2:
  issues.append("No architecture or pedagogical scientific diagram")
 if len(records)!=6:issues.append("Expected six complete non-plot scientific figures")
 result={"status":"PASS_STAGED_FIGURE_PREFLIGHT_NOT_MANUSCRIPT_APPROVAL" if not issues else "HOLD",
     "figures_inspected":len(records),"source_or_model_figure_count":len(records),
     "standalone_numerical_plot_count":0,"figure_records":records,
     "production_approved":False,"issue_count":len(issues),"issues":issues,
     "publication_note":"Current complete PDF/TeX and Figure 6.20 not independently verified"}
 return result
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--stage",required=True,type=Path)
 p.add_argument("--report",type=Path);a=p.parse_args()
 report=check(a.stage)
 out=json.dumps(report,indent=2)+"\n"
 print(out,end="")
 if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(out)
 if report["issue_count"]:raise SystemExit(1)
