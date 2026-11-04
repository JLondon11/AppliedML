"""Check Chapter 5 case-study evidence gates; does not verify neural predictions."""
import csv
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"CH5_CODE_MANIFEST.csv"
rows=list(csv.DictReader(p.open(encoding="utf-8")))
keys=["named_model_real_dataset_inference","complete_executable_implementation",
      "generated_model_output_figure_verified","generated_model_metrics_table_verified",
      "dataset_checkpoint_provenance_complete"]
assert len(rows)==11
assert len({r["id"] for r in rows})==11
for r in rows:
    assert all(r[k] in ("YES","NO") for k in keys)
    if r["qa_status"]=="APPROVED":
        assert all(r[k]=="YES" for k in keys),r["id"]
print("Chapter 5 evidence gate passed; approved:",sum(r["qa_status"]=="APPROVED" for r in rows))
