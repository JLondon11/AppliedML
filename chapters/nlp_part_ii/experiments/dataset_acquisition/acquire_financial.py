#!/usr/bin/env python3
"""Download verified official FinQA and TAT-QA financial document benchmarks."""
import hashlib,json,time,argparse
from pathlib import Path
import requests
FILES={
 "finqa_train":("https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json","finqa"),
 "finqa_dev":("https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json","finqa"),
 "finqa_test":("https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json","finqa"),
 "tatqa_train":("https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json","tatqa"),
 "tatqa_dev":("https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json","tatqa"),
 "tatqa_test":("https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_test_gold.json","tatqa"),
}

def grab(url,path):
    for attempt in range(3):
        try:
            response=requests.get(url,timeout=(15,120))
            response.raise_for_status()
            path.write_bytes(response.content)
            if len(response.content)<10000:raise ValueError("Unexpectedly small dataset")
            return
        except Exception:
            path.unlink(missing_ok=True)
            if attempt==2:raise
            time.sleep(attempt+2)

def verify(path,kind):
    data=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data,list) or len(data)<100:raise ValueError("Expected substantial list of labeled items")
    if kind=="finqa":
        if not all(isinstance(x,dict) and x.get("qa",{}).get("question") and
                   x.get("qa",{}).get("answer") is not None and
                   ("table" in x or "pre_text" in x) for x in data):
            raise ValueError("FinQA gold questions and evidence missing")
    else:
        if not all(isinstance(x,dict) and ("table" in x or "paragraphs" in x) and
                   isinstance(x.get("questions"),list) for x in data):
            raise ValueError("TAT-QA document/question schema mismatch")
        if sum(len(x["questions"]) for x in data)<100:
            raise ValueError("TAT-QA too few questions")
    return {"documents":len(data),
            "questions":sum(len(x.get("questions",[])) for x in data)
                        if kind=="tatqa" else len(data)}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,(url,kind) in FILES.items():
        path=args.output/(name+".json")
        try:
            if not path.exists():grab(url,path)
            metadata=verify(path,kind)
            row={"dataset":name,"status":"acquired_verified","url":url,
                 "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                 "bytes":path.stat().st_size,**metadata}
        except Exception as error:
            row={"dataset":name,"status":"blocked","error":str(error),"url":url}
        records.append(row)
        print(json.dumps(row),flush=True)
    (args.output/"manifest.json").write_text(json.dumps(records,indent=2)+"\n")
    if any(row["status"]!="acquired_verified" for row in records):raise SystemExit(2)

if __name__=="__main__":main()
