#!/usr/bin/env python3
"""Acquire 1,200 public DocVQA examples with images/OCR/reference answers.

Source: nielsr/docvqa_1200_examples, documented by Hugging Face Transformers.
Uses train=1000, test=200; validation is grouped out of TRAIN images only.
Real document images and human answer annotations; NOT official full DocVQA.
"""
import argparse,hashlib,json,random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
 p.add_argument("--seed",type=int,default=42)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 ds=load_dataset("nielsr/docvqa_1200_examples")
 if len(ds["train"])!=1000 or len(ds["test"])!=200:
  raise ValueError("Expected 1000/200 sample; dataset changed")
 dataset={}
 corrected_boxes=0
 for split in ("train","test"):
  rows=[]
  for i,r in enumerate(ds[split]):
   image=r["image"]
   if not isinstance(image,Image.Image):raise TypeError(f"{split}[{i}] not PIL image")
   rgb=image.convert("RGB")
   image_sha=hashlib.sha256(rgb.tobytes()+str(rgb.size).encode()).hexdigest()
   imagepath=a.output/"images"/(image_sha+".png")
   imagepath.parent.mkdir(parents=True,exist_ok=True)
   if not imagepath.exists():rgb.save(imagepath)
   words=r["words"];boxes=r["bounding_boxes"]
   if not words or len(words)!=len(boxes):raise ValueError("Missing/mismatched OCR")
   normalized=[]
   for b in boxes:
    if not isinstance(b,(tuple,list)) or len(b)!=4:raise ValueError("Malformed bounding box")
    bb=[int(round(float(v))) for v in b]
    if not all(0<=x<=1000 for x in bb):
     raise ValueError("Out-of-range normalized bbox "+repr(bb))
    if bb[0]>bb[2] or bb[1]>bb[3]:
     corrected_boxes+=1
     bb=[min(bb[0],bb[2]),min(bb[1],bb[3]),max(bb[0],bb[2]),max(bb[1],bb[3])]
    normalized.append(bb)
   answers=r["answers"]
   if isinstance(answers,str):answers=[answers]
   if not answers:raise ValueError("No answers")
   rows.append({"id":f"{split}_{r['id']}","document_id":image_sha,
    "image":str(imagepath.resolve()),"question":r["query"],"words":words,
    "boxes":normalized,"answers":answers})
  dataset[split]=rows
 train_ids={r["document_id"] for r in dataset["train"]}
 test_ids={r["document_id"] for r in dataset["test"]}
 overlap=train_ids & test_ids
 if overlap:
  raise ValueError(f"Train/test document overlap (images): {len(overlap)}")
 # Grouped split: a document and all its QAs remain in the same split.
 all_docs=sorted(train_ids);rng=random.Random(a.seed);rng.shuffle(all_docs)
 val_ids=set(all_docs[:max(1,round(len(all_docs)*.1))])
 splits={"train":[r for r in dataset["train"] if r["document_id"] not in val_ids],
  "validation":[r for r in dataset["train"] if r["document_id"] in val_ids],
  "test":dataset["test"]}
 if not splits["validation"] or not splits["train"]:raise ValueError("Empty grouped split")
 for split,rows in splits.items():
  with (a.output/(split+".jsonl")).open("w") as file:
   for r in rows:file.write(json.dumps(r,ensure_ascii=False)+"\n")
 output={"dataset":"nielsr/docvqa_1200_examples",
   "origin":"Public subset of original DocVQA, not full official benchmark",
   "question_counts":{s:len(v) for s,v in splits.items()},
   "document_counts":{s:len({r["document_id"] for r in v}) for s,v in splits.items()},
   "train_validation_test_doc_disjoint":True,"seed":a.seed,
   "ocr_bbox_corner_reordering_count":corrected_boxes,
   "downloaded_dataset_splits":{"train":1000,"test":200},
   "limitations":["Not full DocVQA","Eval only on 200 provided test items",
      "Acquisition checks OCR and image overlap but does not execute a model"]}
 (a.output/"manifest.json").write_text(json.dumps(output,indent=2)+"\n")
 print(json.dumps(output,indent=2),flush=True)
if __name__=="__main__":main()
