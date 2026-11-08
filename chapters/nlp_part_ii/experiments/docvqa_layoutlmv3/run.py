#!/usr/bin/env python3
"""Train/evaluate LayoutLMv3 extractive QA on real document images and OCR.

JSONL split schema: id, document_id, image, question, words, boxes, answers.
Boxes are integer OCR coordinates on the LayoutLMv3 0..1000 scale.
Splits MUST have disjoint document_ids, even when question ids differ.
Requires real train/validation/test manifests and accessible checkpoints.
"""
import argparse,hashlib,json,random,re
from pathlib import Path

def normalize(x):
    return " ".join(re.findall(r"\w+",str(x).casefold(),flags=re.UNICODE))

def anls(reference,prediction):
    a=normalize(reference);b=normalize(prediction)
    if not a or not b:return float(a==b)
    prior=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        curr=[i]
        for j,y in enumerate(b,1):
            curr.append(min(prior[j]+1,curr[-1]+1,prior[j-1]+(x!=y)))
        prior=curr
    score=1-prior[-1]/max(len(a),len(b))
    return score if score>=.5 else 0.

def answer_word_span(words,answers):
    for answer in answers:
        for start in range(len(words)):
            for end in range(start+1,min(start+19,len(words)+1)):
                if normalize(" ".join(words[start:end]))==normalize(answer) and normalize(answer):
                    return (start,end-1)
    return None

def load_manifest(path):
    records=[]
    for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        item=json.loads(line)
        required=("id","document_id","image","question","words","boxes","answers")
        if any(field not in item for field in required):
            raise ValueError(f"{path}:{line_no} missing field(s)")
        if not isinstance(item["words"],list) or len(item["words"])!=len(item["boxes"]) or not item["words"]:
            raise ValueError(f"{path}:{line_no} OCR mismatch")
        if not item["answers"] or not isinstance(item["answers"],list):
            raise ValueError(f"{path}:{line_no} missing reference answers")
        for b in item["boxes"]:
            if not (isinstance(b,list) and len(b)==4 and
                    all(type(v) is int and 0<=v<=1000 for v in b) and
                    b[0]<=b[2] and b[1]<=b[3]):
                raise ValueError(f"{path}:{line_no} invalid OCR box")
        image=Path(item["image"])
        if not image.is_absolute():image=path.parent/image
        if not image.is_file():raise FileNotFoundError(image)
        item["image"]=str(image.resolve())
        records.append(item)
    if not records or len({str(x["id"]) for x in records})!=len(records):
        raise ValueError("Empty manifest or duplicate IDs")
    return records

def assert_disjoint(splits):
    for k,a in splits.items():
        for j,b in splits.items():
            if k>=j:continue
            if {str(x["id"]) for x in a}&{str(x["id"]) for x in b}:
                raise ValueError(f"Question leakage: {k}/{j}")
            if {str(x["document_id"]) for x in a}&{str(x["document_id"]) for x in b}:
                raise ValueError(f"Document leakage: {k}/{j}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ("train","validation","test"):
        p.add_argument("--"+x,type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",default="microsoft/layoutlmv3-base")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--learning-rate",type=float,default=2e-5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--local-only",action="store_true")
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1:p.error("Positive epochs and batch size required")
    splits={k:load_manifest(getattr(a,k)) for k in ("train","validation","test")}
    assert_disjoint(splits)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import LayoutLMv3Processor,LayoutLMv3ForQuestionAnswering
    from torch.utils.data import Dataset,DataLoader,Subset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    processor=LayoutLMv3Processor.from_pretrained(a.checkpoint,apply_ocr=False,
                                                 local_files_only=a.local_only)
    model=LayoutLMv3ForQuestionAnswering.from_pretrained(a.checkpoint,
                                                        local_files_only=a.local_only)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.to(device)
    class Examples(Dataset):
        def __init__(self,rows):self.rows=rows
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            r=self.rows[i]
            with Image.open(r["image"]) as image:
                enc=processor(image.convert("RGB"),r["question"],r["words"],
                              boxes=r["boxes"],max_length=512,
                              truncation=True,padding="max_length",return_tensors="pt")
            words=enc.word_ids(batch_index=0);sequence=enc.sequence_ids(batch_index=0)
            positions=[w if s==1 else None for w,s in zip(words,sequence)]
            span=answer_word_span(r["words"],r["answers"])
            start=end=0;aligned=False
            if span is not None:
                matches=[j for j,w in enumerate(positions)
                         if w is not None and span[0]<=w<=span[1]]
                if matches and positions[matches[0]]==span[0] and positions[matches[-1]]==span[1]:
                    start,end=matches[0],matches[-1];aligned=True
            e={k:v.squeeze(0) for k,v in enc.items() if isinstance(v,torch.Tensor)}
            e["start_positions"]=torch.tensor(start)
            e["end_positions"]=torch.tensor(end)
            return e,aligned,positions
    ds={k:Examples(v) for k,v in splits.items()}
    aligned=[i for i in range(len(ds["train"])) if ds["train"][i][1]]
    if len(aligned)<2:raise RuntimeError("Too few OCR-aligned answers; refusing to train")
    def collate(items):
        return {k:torch.stack([row[0][k] for row in items]) for k in items[0][0]}
    train_dl=DataLoader(Subset(ds["train"],aligned),batch_size=a.batch_size,
                        shuffle=True,collate_fn=collate)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate)
    def evaluate(key):
        model.eval();results=[]
        with torch.no_grad():
            for i,r in enumerate(splits[key]):
                inputs,_,positions=ds[key][i]
                model_inputs={k:v.unsqueeze(0).to(device) for k,v in inputs.items()
                              if k not in ("start_positions","end_positions")}
                pred=model(**model_inputs)
                starts=pred.start_logits[0];ends=pred.end_logits[0]
                valid=[j for j,w in enumerate(positions) if w is not None]
                best=(-float("inf"),None,None)
                for st in valid:
                    for en in valid:
                        if en<st or en>=st+30:continue
                        score=float(starts[st]+ends[en])
                        if score>best[0]:best=(score,st,en)
                guess="" if best[1] is None else " ".join(
                    r["words"][positions[best[1]]:positions[best[2]]+1])
                results.append({"id":r["id"],"dataset":"DocVQA manuscript inputs",
                                "split":key,"model":a.checkpoint,"version":"fine-tuned",
                                "reference":r["answers"][0],"all_references":r["answers"],
                                "prediction":guess,
                                "anls":max(anls(x,guess) for x in r["answers"]),
                                "exact_match":int(any(normalize(x)==normalize(guess)
                                                     for x in r["answers"]))})
        return results
    history=[];best=-1;best_state=None
    for epoch in range(1,a.epochs+1):
        model.train();total=0;n=0
        for item in train_dl:
            item={k:v.to(device) for k,v in item.items()}
            optimizer.zero_grad(set_to_none=True);loss=model(**item).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            count=len(item["start_positions"]);total+=float(loss.item())*count;n+=count
        val=evaluate("validation")
        accuracy=sum(r["anls"] for r in val)/len(val)
        history.append({"epoch":epoch,"train_loss":total/n,"validation_anls":accuracy})
        print(json.dumps(history[-1]),flush=True)
        if accuracy>best:
            best=accuracy
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test=evaluate("test")
    metrics={"anls":sum(r["anls"] for r in test)/len(test),
             "exact_match":sum(r["exact_match"] for r in test)/len(test)}
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"predictions_test.jsonl").open("w") as f:
        for row in test:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    result={"experiment":"layoutlmv3_docvqa","checkpoint":a.checkpoint,
            "metrics":metrics,"sizes":{k:len(v) for k,v in splits.items()},
            "train_ocr_aligned":len(aligned),
            "train_dropped_unaligned":len(ds["train"])-len(aligned),
            "manifest_sha256":{k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest()
                               for k in splits},"history":history,"seed":a.seed}
    (a.output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([r["epoch"] for r in history],
            [r["validation_anls"] for r in history],marker="o")
    ax.set(xlabel="Epoch",ylabel="Validation ANLS",ylim=(0,1))
    fig.tight_layout();fig.savefig(a.output/"validation_curve.pdf");plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
