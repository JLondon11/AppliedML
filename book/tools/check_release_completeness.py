#!/usr/bin/env python3
from pathlib import Path
import csv, json, sys

ROOT=Path(__file__).resolve().parents[2]
CHAPTERS=[
"01_foundations_machine_learning","02_deep_learning_part_i","03_deep_learning_part_ii",
"04_computer_vision_part_i","05_computer_vision_part_ii","06_nlp_part_i","07_nlp_part_ii",
"08_generative_ai_part_i","09_generative_ai_part_ii","10_reinforcement_learning",
"11_scientific_ai","12_systems_engineering_mlops"
]
REQUIRED=[
"manuscript/chapter.tex",
"manuscript/chapter.pdf",
"figures/FIGURE_MANIFEST.csv",
"code/CODE_INVENTORY.csv",
"code/CODE_MANIFEST.csv",
"code/REPRODUCIBILITY.md",
"instructor/solutions.tex",
"instructor/solutions.pdf",
"qa/CHAPTER_QA.md",
"qa/CODE_AUDIT.csv",
]
rows=[]
for ch in CHAPTERS:
    base=ROOT/"book"/"chapters"/ch
    for rel in REQUIRED:
        p=base/rel
        rows.append({"chapter":ch,"required_path":rel,"present":p.exists(),"size_bytes":p.stat().st_size if p.exists() else 0})
qa=ROOT/"book"/"qa"; qa.mkdir(parents=True,exist_ok=True)
with (qa/"RELEASE_COMPLETENESS.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
missing=[r for r in rows if not r["present"]]
summary={"required_items":len(rows),"present":len(rows)-len(missing),"missing":len(missing)}
(qa/"RELEASE_COMPLETENESS.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
if missing:
    print("\nMissing canonical release artifacts:")
    for r in missing: print(f"- {r['chapter']}: {r['required_path']}")
    sys.exit(3)
