"""Rebuild all seven Chapter 6 source/model figures and test actual pixel parity.

Requires the self-contained Pass64 evidence package with original executed
checkpoint, authentic-source scripts, and model architecture. This tests
figure science/production staging, not the unavailable final chapter PDF.
"""
import argparse,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import fitz,numpy as np

FILES=["tissue_source_compact.pdf","digits_source_compact.pdf",
       "nci_source_compact.pdf","tissue_vae_output_compact_1.pdf",
       "tissue_vae_output_compact_2.pdf",
       "smiles_vae_architecture_compact.pdf","tissue_vae_architecture_compact.pdf"]
PANELS=[7,20,6,16,16,0,0]
SOURCE_SHA="c5b3ef509a92f16d4c29be8cf0300fe75d53e13a3ce650159db932caea8dcc1b"
CHECKPOINT_SHA="204d5bd26a3361584285f108c7e74b9bdad2e5ee9d52310c7a5204f12d08bbbe"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inspect_pdf(p):
    with fitz.open(p) as doc:
        if len(doc)!=1:raise ValueError("Expected one page: "+str(p))
        page=doc[0]
        rendered=page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
        pixels=np.frombuffer(rendered.samples,dtype=np.uint8).copy()
        spans=[s for b in page.get_text("dict")["blocks"]
             for line in b.get("lines",[])
             for s in line.get("spans",[]) if s["text"].strip()]
        if not spans:raise ValueError("No selectable source caption: "+str(p))
        off=[s["text"][:40] for s in spans
             if s["bbox"][0]<-.1 or s["bbox"][1]<-.1
             or s["bbox"][2]>page.rect.width+.1
             or s["bbox"][3]>page.rect.height+.1]
        return ((rendered.width,rendered.height),pixels,
                page.rect.width,min(s["size"] for s in spans),
                len(page.get_images(full=True)),off)

def verify(bundle,checkpoint):
    bundle=Path(bundle).resolve()
    checkpoint=Path(checkpoint).resolve()
    if sha(checkpoint)!=CHECKPOINT_SHA:raise ValueError("Original checkpoint identity changed")
    scripts=["build_compact_real_image_plates.py","print_width_architecture_plates.py",
             "audit_true_tissue_vae_vs_nearest.py"]
    for name in scripts:
        if not (bundle/name).is_file():
            raise FileNotFoundError("Missing exact source dependency in delivery: "+name)
    output=[]
    with tempfile.TemporaryDirectory(prefix="chapter6_original_") as temp:
        temp=Path(temp)
        subprocess.run([sys.executable,str(bundle/scripts[0]),"--checkpoint",
            str(checkpoint),"--output",str(temp)],cwd=bundle,check=True,capture_output=True,text=True)
        subprocess.run([sys.executable,str(bundle/scripts[1]),"--output",
            str(temp)],cwd=bundle,check=True,capture_output=True,text=True)
        provenance=json.loads((temp/"print_compact_provenance.json").read_text())
        if provenance["source_photo_sha256"]!=SOURCE_SHA or provenance["checkpoint_sha256"]!=CHECKPOINT_SHA:
            raise ValueError("Original microscopy or checkpoint SHA mismatch")
        for name,expected in zip(FILES,PANELS):
            old=bundle/"figures"/name;new=temp/name
            a,b=inspect_pdf(old),inspect_pdf(new)
            unequal=int(np.count_nonzero(a[1]!=b[1])) if a[0]==b[0] else -1
            if unequal:raise ValueError(f"Nonidentical original figure pixel buffer: {name}: {unequal}")
            if abs(b[2]-468)>.1 or b[3]<7.5 or b[4]!=expected or b[5]:
                raise ValueError("Staged figure physical print QA fails: "+name)
            output.append({"figure":name,"original_pdf_sha256":sha(old),
                 "rendered_channel_differences":unequal,"source_model_panels":b[4],
                 "minimum_original_font_pt":round(b[3],2)})
    return {"status":"PASS_SELF_CONTAINED_FIGURE_REPLAY",
            "source_images_are_authentic":True,
            "seven_figures_pixel_identical":len(output)==7,
            "figures":output,
            "original_photo_sha256":SOURCE_SHA,
            "trained_checkpoint_sha256":CHECKPOINT_SHA,
            "full_manuscript_scientific_and_production_certified":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",required=True,type=Path)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--report",type=Path)
    a=p.parse_args()
    res=verify(a.bundle,a.checkpoint)
    payload=json.dumps(res,indent=2)+"\n"
    print(payload,end="")
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(payload)
