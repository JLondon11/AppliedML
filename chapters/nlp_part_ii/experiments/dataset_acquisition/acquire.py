#!/usr/bin/env python3
"""Fetch and validate public Chapter 9 benchmark files; fail closed."""
import argparse, hashlib, json, tarfile, time
from pathlib import Path
import requests

SOURCES={
 "scifact":"https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
 "cnn_dailymail_test":"https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0/test-00000-of-00001.parquet",
}

def download(url, output):
    tmp=output.with_name(output.name+".partial")
    for attempt in range(3):
        try:
            with requests.get(url,stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:f.write(chunk)
            if tmp.stat().st_size<1000:raise ValueError("Truncated data")
            tmp.rename(output)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt==2:raise
            time.sleep(3*(attempt+1))

def validate_imdb(file):
    counter={f"{s}_{label}":0 for s in ["train","test"] for label in ["pos","neg"]}
    with tarfile.open(file,"r:gz") as tar:
        for member in tar:
            name=member.name.lstrip("./")
            for key in counter:
                split,label=key.split("_")
                if name.startswith(f"aclImdb/{split}/{label}/") and name.endswith(".txt"):
                    counter[key]+=1
    assert all(x==12500 for x in counter.values()),counter
    return counter

def validate_scifact(file):
    counts={}
    expected={"corpus.jsonl","claims_train.jsonl","claims_dev.jsonl","claims_test.jsonl"}
    with tarfile.open(file,"r:gz") as tar:
        for member in tar:
            name=member.name.split("/")[-1]
            if name in expected and member.isfile():
                n=0
                for line in tar.extractfile(member):
                    if line.strip():json.loads(line);n+=1
                counts[name]=n
    assert set(counts)==expected and all(x>=100 for x in counts.values()),counts
    return counts

def validate_cnn(file):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(file)
    assert pf.metadata.num_rows>5000 and {"article","highlights"}<=set(pf.schema_arrow.names)
    return {"rows":pf.metadata.num_rows,"columns":pf.schema_arrow.names}

def validate_wmt(file):
    num=0
    with file.open() as f:
        for line in f:
            item=json.loads(line);assert item["en"] and item["de"]
            num+=1
    assert num>1000
    return {"rows":num}

def get_wmt(out):
    from datasets import load_dataset
    ds=load_dataset("wmt/wmt14","de-en",split="test")
    path=out/"wmt14_de_en_test.jsonl"
    with path.open("w") as f:
        for row in ds:f.write(json.dumps(row["translation"],ensure_ascii=False)+"\n")
    return path,validate_wmt(path)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

IMDB_PARQUET={
 "train":("https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet",
          "db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c"),
 "test":("https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet",
         "b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da"),
}

def acquire_imdb_parquet(out):
    import pyarrow.parquet as pq
    parts=[]
    counts={}
    for split,(url,expected_sha) in IMDB_PARQUET.items():
        file=out/("imdb_"+split+".parquet")
        if not file.exists():download(url,file)
        actual_sha=sha(file)
        if actual_sha!=expected_sha:raise ValueError(f"Official IMDb {split} hash mismatch: {actual_sha}")
        table=pq.read_table(file,columns=["label"])
        labels=table.column("label").to_pylist()
        distribution={str(k):labels.count(k) for k in [0,1]}
        if len(labels)!=25000 or distribution!={"0":12500,"1":12500}:
            raise ValueError(f"IMDb {split} official split mismatch: {distribution}")
        parts.append(str(file))
        counts[split]={"rows":len(labels),"labels":distribution,"sha256":actual_sha}
    return {"files":parts,"counts":counts}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--datasets",nargs="+",default=list(SOURCES)+["wmt14"])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in args.datasets:
        try:
            if name=="imdb":
                validation=acquire_imdb_parquet(args.output)
                file=args.output/"imdb_train.parquet"
                source="huggingface:stanfordnlp/imdb plain_text train/test Parquet, pinned SHA256"
            elif name=="wmt14":
                file,validation=get_wmt(args.output)
                source="huggingface:wmt/wmt14 de-en test"
            else:
                source=SOURCES[name]
                ext=".parquet" if name=="cnn_dailymail_test" else ".tar.gz"
                file=args.output/(name+ext)
                if not file.exists():download(source,file)
                validation={"imdb":validate_imdb,"scifact":validate_scifact,
                            "cnn_dailymail_test":validate_cnn}[name](file)
            row={"name":name,"status":"acquired_verified","url":source,
                 "path":str(file),"bytes":file.stat().st_size,"sha256":sha(file),
                 "validation":validation}
        except Exception as error:
            row={"name":name,"status":"blocked","error":str(error)}
        records.append(row)
        print(json.dumps(row),flush=True)
    records.extend([{"name":"docvqa","status":"portal_login_required",
                     "url":"https://site.docvqa.org/datasets"},
                    {"name":"enterprise","status":"requires_authorized_corpus"}])
    (args.output/"acquisition_manifest.json").write_text(json.dumps(records,indent=2)+"\n")
    if any(x["status"]=="blocked" for x in records):raise SystemExit(2)

if __name__=="__main__":main()
