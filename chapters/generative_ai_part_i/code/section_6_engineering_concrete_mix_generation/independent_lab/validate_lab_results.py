"""Fail-closed completeness audit for real independent concrete strength testing.

This does not authenticate laboratory signatures or certify experimental validity;
a qualified external reviewer must independently inspect original signed records.
No generated/placeholder values can count as measured strengths.
"""
import argparse,csv,hashlib,json,math,datetime
from collections import defaultdict
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--recipes",type=Path,default=Path(__file__).parent/"prospective_mix_shortlist.csv")
p.add_argument("--results",type=Path,default=Path(__file__).parent/"actual_lab_results_TEMPLATE.csv")
p.add_argument("--report",type=Path,default=Path(__file__).parent/"external_signed_lab_certificate.pdf")
p.add_argument("--out",type=Path,default=Path(__file__).parent/"lab_verification.json")
args=p.parse_args()
if not args.recipes.is_file():raise SystemExit("HOLD: original candidate provenance missing.")
recipes=list(csv.DictReader(args.recipes.open(newline="")))
assert len(recipes)==6 and len({r["blind_id"] for r in recipes})==6
if not args.report.is_file() or args.report.stat().st_size<5000:
    raise SystemExit("HOLD: independent laboratory report absent or incomplete; NO validation.")
if args.report.open("rb").read(5)!=b"%PDF-":
    raise SystemExit("HOLD: submitted report is not a PDF certificate.")
if not args.results.is_file():raise SystemExit("HOLD: original measured specimen CSV missing.")
records=list(csv.DictReader(args.results.open(newline="")))
if not records:raise SystemExit("HOLD: no actual laboratory specimen measurements supplied.")
required={"blind_id","lab_batch_id","cylinder_id","test_date","cure_age_days",
          "measured_strength_mpa","laboratory","report_id"}
if not required.issubset(records[0]):raise SystemExit("HOLD: missing signed-data provenance fields.")
arms={r["blind_id"]:r["arm"] for r in recipes}
by=defaultdict(list);unique=set();labnames=set();reportids=set()
for r in records:
    if r["blind_id"] not in arms:raise ValueError("Unknown blinded formula ID.")
    if not r["lab_batch_id"] or not r["cylinder_id"] or not r["laboratory"] or not r["report_id"]:
        raise ValueError("Missing lab/batch/specimen/report provenance.")
    key=(r["lab_batch_id"],r["cylinder_id"])
    if key in unique:raise ValueError("Duplicate physical cylinder recorded.")
    unique.add(key);labnames.add(r["laboratory"]);reportids.add(r["report_id"])
    try:
        strength=float(r["measured_strength_mpa"]);age=float(r["cure_age_days"])
        datetime.date.fromisoformat(r["test_date"])
    except (ValueError,TypeError):
        raise ValueError("Invalid actual test date, strength or curing age.")
    if not math.isfinite(strength) or strength<=0 or age!=28:
        raise ValueError("Invalid or non-28-day physical strength measurement.")
    by[r["blind_id"],r["lab_batch_id"]].append(strength)
for r in recipes:
    batches=[v for (rid,bid),v in by.items() if rid==r["blind_id"]]
    if len(batches)<3 or any(len(v)<3 for v in batches):
        raise SystemExit(f"HOLD: {r['blind_id']} needs >=3 independently mixed batches, >=3 cylinders per batch.")
summary={}
for rid,arm in arms.items():
    batchmeans=[float(np.mean(vals)) for (bid,batchid),vals in by.items() if bid==rid]
    summary[rid]={"arm":arm,"n_independent_batches":len(batchmeans),
                  "batch_mean_strength_mpa":batchmeans,
                  "overall_batch_mean_mpa":float(np.mean(batchmeans))}
report={"status":"RECORD_COMPLETENESS_ONLY_MANUAL_EXTERNAL_VERIFICATION_REQUIRED",
        "report_sha256":hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "original_lab_csv_sha256":hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "source_candidate_recipe_sha256":hashlib.sha256(args.recipes.read_bytes()).hexdigest(),
        "reported_laboratories":sorted(labnames),"lab_report_ids":sorted(reportids),
        "n_physical_cylinders":len(records),"n_unique_batch_codes":len({k[1] for k in by}),
        "summary":summary,
        "limitations":["Cannot authenticate PDF signature cryptographically",
                       "Cannot verify machine calibration, wet-lab operations, or actual mix batching",
                       "Manual match of the physical lab certificate to CSV entries required",
                       "No causal interpretation without checking mixture composition and blinding"]}
args.out.write_text(json.dumps(report,indent=2)+"\n")
print("Record completeness checks passed. DO NOT certify until independent lab report and raw measurement audit.")
