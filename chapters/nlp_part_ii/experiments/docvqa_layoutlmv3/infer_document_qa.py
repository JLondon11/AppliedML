#!/usr/bin/env python3
"""OCR-grounded LayoutLMv3 document QA inference on real page images.

Requires a locally fine-tuned LayoutLMv3ForQuestionAnswering checkpoint WITH
the saved LayoutLMv3Processor, plus JSON OCR annotations: {"words": [str],
"boxes": [[x0,y0,x1,y1]]}. Box coordinates must be normalized to 0-1000.

Not a benchmark, not an executed model result, and not suitable for a base-only
checkpoint with randomly initialized QA weights. For evaluation, compare saved
answers with real held-out DocVQA references using ANLS.
"""
import argparse
import json
from pathlib import Path

def infer(checkpoint: Path, image_path: Path, ocr_path: Path, question: str):
    import torch
    from PIL import Image
    from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
    if not checkpoint.is_dir():
        raise FileNotFoundError("An existing fine-tuned QA checkpoint is required")
    record=json.loads(ocr_path.read_text(encoding="utf-8"))
    words,boxes=record["words"],record["boxes"]
    if not words or len(words)!=len(boxes):
        raise ValueError("OCR words and bounding boxes must align")
    for box in boxes:
        if (len(box)!=4 or any(type(v) is not int for v in box)
            or not (0<=box[0]<=box[2]<=1000 and 0<=box[1]<=box[3]<=1000)):
            raise ValueError("Invalid normalized OCR box")
    processor=LayoutLMv3Processor.from_pretrained(checkpoint,apply_ocr=False)
    model=LayoutLMv3ForQuestionAnswering.from_pretrained(checkpoint).eval()
    with Image.open(image_path) as image:
        encoding=processor(image.convert("RGB"),question,words,boxes=boxes,
                           truncation=True,max_length=512,return_tensors="pt")
    with torch.inference_mode():
        output=model(**encoding)
    word_ids=encoding.word_ids(batch_index=0)
    segments=encoding.sequence_ids(batch_index=0)
    valid=[i for i,(w,seg) in enumerate(zip(word_ids,segments))
           if w is not None and seg==1]
    best=(-float("inf"),None,None)
    for start in valid:
        for end in valid:
            if end<start or end-start>25:
                continue
            score=float(output.start_logits[0,start]+output.end_logits[0,end])
            if score>best[0]:
                best=(score,start,end)
    if best[1] is None:
        raise RuntimeError("No OCR-aligned answer tokens are available")
    first,last=word_ids[best[1]],word_ids[best[2]]
    return {"answer":" ".join(words[first:last+1]),"word_span":[first,last],
            "page_image":str(image_path),"checkpoint":str(checkpoint)}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--ocr",type=Path,required=True)
    p.add_argument("--question",required=True)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    result=infer(a.checkpoint,a.image,a.ocr,a.question)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":
    main()
