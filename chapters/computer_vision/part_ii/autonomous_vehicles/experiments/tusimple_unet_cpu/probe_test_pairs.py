from huggingface_hub import HfApi,hf_hub_download
import json
repo="dhbloo/TuSimple"
api=HfApi()
entries={e.path for e in api.list_repo_tree(repo,repo_type="dataset",recursive=True)}
test=hf_hub_download(repo_id=repo,repo_type="dataset",filename="test_label.json")
with open(test) as f: records=[json.loads(row) for row in f if row.strip()]
print("TEST_ANNOTATIONS",len(records))
for r in records[:8]:
    path=r["raw_file"]
    print("SAMPLE",path,"HAS_IMAGE",path in entries,"lanes",len(r["lanes"]))
matched=[r for r in records if r["raw_file"] in entries]
print("MATCHED_TEST_IMAGES",len(matched))
print("EXAMPLE_MATCHES",[r["raw_file"] for r in matched[:8]])
print("OTHER_LABELS",sorted(s for s in entries if "label" in s.lower())[:20])
