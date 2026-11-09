#!/usr/bin/env python3
"""Chapter 9 experiment execution-readiness inventory; never auto-certifies.

Usage: python audit_execution_readiness.py --evidence-root outputs --report readiness.json
The output is a planning/QA gate, not a substitute for running named models.
"""
import argparse, hashlib, importlib.util, json
from pathlib import Path

UNITS = {
    "9.6":("summarization","BART / CNN-DailyMail"),
    "9.10":("sentiment_bert","BERT / IMDb"),
    "9.12":("translation","MarianMT / WMT14"),
    "9.17":("research_agent","Research agent / SciFact or GAIA"),
    "9.20":("enterprise_rag","RAG / BEIR or authorized enterprise corpus"),
    "9.24":("document_intelligence","LayoutLMv3 / DocVQA"),
    "9.26":("scientific_literature","Literature RAG / SciFact"),
    "9.28":("enterprise_knowledge","Enterprise assistant / authorized corpus"),
}
REQUIRED = ("unit_manifest.json","dataset_manifest.json","model_manifest.json",
            "heldout_predictions.jsonl","evaluation_metrics.json","run_command.txt",
            "execution.log","experiment.py","measured_results.pdf","per_sample_evaluation.csv")

def audit(evidence_root: Path):
    results = []
    for section,(slug,model) in UNITS.items():
        folder=evidence_root/slug
        missing=[name for name in REQUIRED if not (folder/name).is_file()
                 or (folder/name).stat().st_size == 0]
        results.append({"section":section,"unit":slug,"expected_model_and_data":model,
                        "status":"BLOCKED_MISSING_EVIDENCE" if missing else "REVIEW_REQUIRED",
                        "missing_files":missing,"certified":False})
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    rows=audit(args.evidence_root)
    summary={"total":8,"automatically_certified":0,
             "blocked":sum(bool(x["missing_files"]) for x in rows),
             "review_required":sum(not x["missing_files"] for x in rows),
             "dependencies_missing":[p for p in ("transformers","datasets","rouge_score","sacrebleu")
                                     if importlib.util.find_spec(p) is None],
             "warning":"Artifact presence alone NEVER certifies experimental validity.",
             "units":rows}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
