"""Independent source and executed-model integrity audit for Chapter 6 digit galleries.

The input is a trusted original GitHub Actions artifact from executed run
44209629586 and the figures produced by the pinned source generator.
This is a source-specific scientific audit, NOT a manuscript publication gate.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
import fitz
from sklearn.datasets import load_digits

FILES={
 "predictions.csv":"de376724ba102a09f67e75208aa90dd57b79d2caf81942f19af421feaae4c2d3",
 "generated_conditional.pt":"bebaaa56004c1180fc0e6daa09198f5c74e8a527e6cdb4d149ed943eddfe0645",
 "generated_unconditional.pt":"70f93ba0b05d61965df45a2bf24afb61c4785d8569b94ee8210e42fe19acf708",
 "split_indices.pt":"283aa97a11a75a96b7e0f8e57c6101c1ee58e3bb36f3a4b4d593fba549b11974",
 "manifest.json":"1a6ef2de4fbc56245aea3a919f59160b362d5b5d52d5021860ee30c10a381ba4",
}
ORIGINAL="f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443"
PDFS={
 "actual_original_vs_diffusion_digits_1.pdf":"9cefed2b47fb80053aee0688a3eee090921e26065efd46fdf197e1f4e3b628a5",
 "actual_original_vs_diffusion_digits_2.pdf":"0a94bd571d76d253c2bacc1c99c05b974a9d3b48bbc5b8799ea271e2dbf81939",
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(source,figures):
    source=Path(source);figures=Path(figures)
    for name,expected in FILES.items():
        if sha(source/name)!=expected:raise ValueError("Recorded successful run source mismatch: "+name)
    dataset=load_digits();y=dataset.target.astype("int64")
    if hashlib.sha256(dataset.data.tobytes()+y.tobytes()).hexdigest()!=ORIGINAL:
        raise ValueError("The original 1797 real scan source changed")
    # Unpickling permitted only after validating actual original self-generated
    # archive member hashes; do not pass unrelated or untrusted files here.
    split=torch.load(source/"split_indices.pt",map_location="cpu",weights_only=False)
    sets={k:set(map(int,split[k])) for k in ("train","validation","test")}
    if [len(sets[k]) for k in ("train","validation","test")]!=[1257,270,270]:
        raise ValueError("Original 1257/270/270 source partition invalid")
    if sets["train"]&sets["test"] or sets["train"]&sets["validation"] or sets["validation"]&sets["test"]:
        raise ValueError("Original real image leakage across model partitions")
    tensors={}
    for arm in ("conditional","unconditional"):
        t=torch.load(source/f"generated_{arm}.pt",map_location="cpu",weights_only=False)
        value=t["generated"]
        if hasattr(value,"detach"):value=value.detach().cpu().numpy()
        tensors[arm]=np.asarray(value,dtype=float)
        if tensors[arm].shape!=(1000,64):raise ValueError("Original generated sample size")
        if not np.array_equal(np.asarray(t["prompts"]),np.arange(10).repeat(100)):
            raise ValueError("Original source output prompts missing or altered")
    with (source/"predictions.csv").open(newline="",encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=2000:raise ValueError("Missing original per-image output probability records")
    by={}
    for row in rows:
        arm=row["arm"];idx=int(row["index"])
        if arm not in tensors or idx not in range(1000) or (arm,idx) in by:
            raise ValueError("Duplicate or unknown original output identifier")
        ytrue=int(row["prompt_label"]);pred=int(row["classifier_prediction"])
        if ytrue!=idx//100 or not 0<=pred<=9:raise ValueError("Original label alignment failed")
        probs=np.array([float(row[f"p_{k}"]) for k in range(10)])
        if abs(probs.sum()-1)>1e-5 or int(probs.argmax())!=pred:
            raise ValueError("Original output probabilities inconsistent")
        by[(arm,idx)]=pred
    measured={arm:sum(by[(arm,i)]==i//100 for i in range(1000)) for arm in tensors}
    if measured!={"conditional":779,"unconditional":94}:
        raise ValueError("Recorded successful original source classifier scores drifted")
    meta=json.loads((figures/"actual_digit_image_comparison_provenance.json").read_text())
    if meta["original_source_sha256"]!=ORIGINAL or len(meta["selected_samples"])!=30:
        raise ValueError("The scientific plate source manifest is incomplete")
    if meta["observed_classifier_matches"]!=measured or meta["manuscript_production_approved"]:
        raise ValueError("Incorrect scientific/production figure decision")
    first_in_test={}
    for d in range(10):
        first_in_test[d]=next(int(i) for i in split["test"] if y[int(i)]==d)
    for record in meta["selected_samples"]:
        cls=record["source_class"];kind=record["kind"];position=record["source_position"]
        if not 0<=cls<=9 or kind not in ("original","conditional","unconditional"):
            raise ValueError("Unrecognized figure source/prediction type")
        if kind=="original":
            if position!=first_in_test[cls]:raise ValueError("Source scan selection changed")
            pixels=np.asarray(dataset.images[position]/16,dtype="<f4")
        else:
            if position!=100*cls or record["actual_prediction"]!=by[(kind,position)]:
                raise ValueError("Displayed generator tensor or classifier prediction changed")
            pixels=np.asarray(tensors[kind][position].reshape(8,8),dtype="<f4")
        if hashlib.sha256(pixels.tobytes()).hexdigest()!=record["pixel_sha256"]:
            raise ValueError("Actual figure source/model pixel identity mismatch")
    pdf_rows=[]
    for name,expected in PDFS.items():
        file=figures/name
        if sha(file)!=expected:raise ValueError("Exactly rendered scientific source PDF changed")
        with fitz.open(file) as doc:
            if len(doc)!=1:raise ValueError("Expected one-page textbook graphic")
            p=doc[0]
            spans=[s for b in p.get_text("dict")["blocks"]
                   for line in b.get("lines",[]) for s in line.get("spans",[])
                   if s["text"].strip()]
            clipped=[s["text"][:30] for s in spans if s["bbox"][0]<-.1
                     or s["bbox"][1]<-.1 or s["bbox"][2]>p.rect.width+.1
                     or s["bbox"][3]>p.rect.height+.1]
            if abs(p.rect.width-468)>.1 or len(p.get_images(full=True))!=15:
                raise ValueError("Original figures do not match scientific source image counts")
            if not spans or min(s["size"] for s in spans)<7.5 or clipped:
                raise ValueError("Text unreadable or outside the original PDF page")
            pdf_rows.append({"name":name,"sha256":expected,"original_or_model_panels":15,
                             "minimum_printed_type_pt":round(min(s["size"] for s in spans),2)})
    return {
      "result":"PASS_SOURCE_ORIGINAL_EXECUTED_DIGIT_IMAGE_CERTIFICATION",
      "original_successful_GitHub_run":44209629586,
      "original_GitHub_artifact_id":14384184921,
      "genuine_original_source_scans":1797,
      "original_disjoint_test_scans":270,
      "actual_generated_images":2000,
      "observed_agreement":measured,
      "selected_original_or_model_images_sha_verified":30,
      "scoped_scientific_d12_status":"SCIENTIFIC_CERTIFIED",
      "broader_parent_scientific_status":"NOT_CERTIFIED_BY_THIS_STUDY",
      "printed_manuscript_status":"HOLD_CURRENT_TEX_AND_PDF_UNAVAILABLE",
      "actual_model_result_PDFs":pdf_rows,
      "note":"Classifier agreement on low resolution class-conditioned digit outputs is not new patient data, a natural-image model, or independent seed-level generator performance."
    }
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--figures",required=True,type=Path)
    p.add_argument("--report",type=Path)
    args=p.parse_args()
    result=verify(args.source,args.figures)
    payload=json.dumps(result,indent=2)+"\n"
    print(payload,end="")
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(payload)
