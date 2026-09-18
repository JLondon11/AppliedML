#!/usr/bin/env python3
from __future__ import annotations
import ast, csv, json, py_compile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CHAPTERS={
"01_foundations_machine_learning":"chapters/foundations_machine_learning",
"02_deep_learning_part_i":"chapters/deep_learning/part_i",
"03_deep_learning_part_ii":"chapters/deep_learning/part_ii",
"04_computer_vision_part_i":"chapters/computer_vision/part_i/foundations",
"05_computer_vision_part_ii":"chapters/computer_vision/part_ii/autonomous_vehicles",
"06_nlp_part_i":"chapters/nlp_part_i",
"07_nlp_part_ii":"chapters/nlp_part_ii",
"08_generative_ai_part_i":"chapters/generative_ai_part_i",
"09_generative_ai_part_ii":"chapters/generative_ai_part_ii",
"10_reinforcement_learning":"chapters/reinforcement_learning",
"11_scientific_ai":"chapters/scientific_ai",
"12_systems_engineering_mlops":"chapters/systems_engineering_and_mlops",
}
SEED_MARKERS=("random.seed","np.random.seed","numpy.random.seed","torch.manual_seed","manual_seed_all","random_state","seed=","seed =")
DATA_MARKERS=("read_csv","read_parquet","load_dataset","datasets.","DataLoader","ImageFolder","open(","Path(","url","download","wget","requests.")
OUTPUT_MARKERS=("savefig",".save(","to_csv","to_json","torch.save","write_text","write_bytes")
PROV_MARKERS=("provenance","dataset","source","doi","arxiv","citation","reference","license")

def imports(tree):
    out=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):
            out.update(a.name.split('.')[0] for a in n.names)
        elif isinstance(n,ast.ImportFrom) and n.module:
            out.add(n.module.split('.')[0])
    return sorted(out)

def has_any(text, markers):
    low=text.lower()
    return any(m.lower() in low for m in markers)

rows=[]
for slug,rel in CHAPTERS.items():
    root=ROOT/rel
    files=sorted(root.rglob("*.py")) if root.exists() else []
    for p in files:
        text=p.read_text(errors="ignore")
        syntax_ok=True; err=""
        try:
            py_compile.compile(str(p),doraise=True)
            tree=ast.parse(text)
            imps=imports(tree)
        except Exception as e:
            syntax_ok=False; err=f"{type(e).__name__}: {e}"; imps=[]
        has_seed=has_any(text,SEED_MARKERS)
        has_data=has_any(text,DATA_MARKERS)
        has_output=has_any(text,OUTPUT_MARKERS)
        has_prov=has_any(text,PROV_MARKERS)
        status="REVIEW"
        if syntax_ok and (not has_data or has_prov) and (not has_data or has_seed) and has_output:
            status="STATIC_PASS"
        rows.append({
            "chapter":slug,"path":p.relative_to(ROOT).as_posix(),"syntax_ok":syntax_ok,
            "syntax_error":err,"imports":";".join(imps),"seed_or_determinism_declared":has_seed,
            "data_or_external_input_detected":has_data,"provenance_language_detected":has_prov,
            "output_artifact_detected":has_output,"static_reproducibility_status":status,
        })

qa=ROOT/"book"/"qa"; qa.mkdir(parents=True,exist_ok=True)
fields=list(rows[0]) if rows else ["chapter","path"]
with (qa/"CODE_AUDIT.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

for slug in CHAPTERS:
    out=ROOT/"book"/"chapters"/slug/"code"; out.mkdir(parents=True,exist_ok=True)
    cr=[r for r in rows if r["chapter"]==slug]
    with (out/"CODE_MANIFEST.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(cr)
    total=len(cr); bad=sum(not r["syntax_ok"] for r in cr); review=sum(r["static_reproducibility_status"]!="STATIC_PASS" for r in cr)
    (out/"REPRODUCIBILITY.md").write_text(
        f"# Reproducibility audit — {slug}\n\n"
        f"Python files audited: **{total}**  \n"
        f"Syntax failures: **{bad}**  \n"
        f"Files requiring reproducibility review: **{review}**\n\n"
        "STATIC_PASS is a static check, not proof of full experimental reproduction.\n"
    )

summary={"python_files":len(rows),"syntax_failures":sum(not r["syntax_ok"] for r in rows),
         "static_pass":sum(r["static_reproducibility_status"]=="STATIC_PASS" for r in rows),
         "review":sum(r["static_reproducibility_status"]!="STATIC_PASS" for r in rows)}
(qa/"CODE_AUDIT_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
print("\nPer-chapter audit summary:")
for slug in CHAPTERS:
    cr=[r for r in rows if r["chapter"]==slug]
    bad=[r for r in cr if not r["syntax_ok"]]
    review=[r for r in cr if r["static_reproducibility_status"]!="STATIC_PASS"]
    print(f"- {slug}: files={len(cr)} syntax_failures={len(bad)} static_pass={len(cr)-len(review)} review={len(review)}")
print("\nSyntax failures:")
for r in rows:
    if not r["syntax_ok"]:
        print(f"- {r['path']}: {r['syntax_error']}")
print("\nFiles requiring reproducibility review:")
for r in rows:
    if r["static_reproducibility_status"]!="STATIC_PASS":
        print(f"- {r['path']}: seed={r['seed_or_determinism_declared']} data={r['data_or_external_input_detected']} provenance={r['provenance_language_detected']} output={r['output_artifact_detected']}")
if summary["syntax_failures"]:
    raise SystemExit(2)
