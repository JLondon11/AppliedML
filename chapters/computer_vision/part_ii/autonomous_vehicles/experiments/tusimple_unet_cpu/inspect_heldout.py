"""Identify legally accessible original TuSimple held-out image+label pairs.
This is dataset discovery only, not inference or a benchmark score.
"""
from huggingface_hub import HfApi
import json
api=HfApi()
for repo in ("dhbloo/TuSimple","AhmedNasr5804/Lane-and-eye-detection-dataset"):
    print("REPOSITORY",repo,flush=True)
    try:
        files=list(api.list_repo_tree(repo,repo_type="dataset",recursive=True))
        print("TOTAL",len(files),flush=True)
        for entry in files[:90]:
            print("ENTRY",entry.path,getattr(entry,"size",None),flush=True)
        print("LABELED_PATHS",sum("test" in x.path.lower() for x in files),
              "IMAGE_PATHS",sum(x.path.lower().endswith((".jpg",".png",".zip")) for x in files),flush=True)
    except Exception as err:
        print("ERROR",repr(err),flush=True)
