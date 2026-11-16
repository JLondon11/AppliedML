"""Verify authentic original-digit evaluator control and saved-class reconciliation.

Fails on missing class coverage, changed original split, missing real test samples,
inconsistent confusion matrix, changed per-class metrics, or stronger-than-supported
aggregate claims. Verifies numerical evidence, not generator image realism.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "original_evaluator_control"
def read(name):
    with (EVIDENCE/name).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))
def main():
    records=read("real_evaluator_vs_generated_class.csv")
    cm=read("real_evaluator_confusion.csv")
    m=json.loads((EVIDENCE/"metrics.json").read_text(encoding="utf-8"))
    counts=read("figure_6_44_class_counts_pass27.csv") if (EVIDENCE/"figure_6_44_class_counts_pass27.csv").exists() else None
    with (ROOT/"figure_6_44_class_counts_pass27.csv").open(newline="",encoding="utf-8") as f:
        counts=list(csv.DictReader(f))
    assert len(records)==10 and [int(x["digit"]) for x in records]==list(range(10))
    assert len(cm)==10 and [int(x["observed\\predicted"]) for x in cm]==list(range(10))
    ntest=correct=cond=uncond=0
    for i,r in enumerate(records):
        n=int(r["real_test_n"]);k=int(r["real_test_correct"])
        nc=int(r["conditional_generated_n"]);kc=int(r["conditional_generated_agree"])
        nu=int(r["unconditional_generated_n"]);ku=int(r["unconditional_generated_agree"])
        mat=[int(cm[i][str(j)]) for j in range(10)]
        assert len(mat)==10 and sum(mat)==n and mat[i]==k and n>0
        assert nc==nu==100 and 0<=kc<=nc and 0<=ku<=nu
        assert math.isclose(float(r["real_test_recall"]),k/n,abs_tol=1e-12)
        assert math.isclose(float(r["conditional_generated_agreement"]),kc/nc,abs_tol=1e-12)
        assert math.isclose(float(r["unconditional_generated_agreement"]),ku/nu,abs_tol=1e-12)
        for arm,x in (("conditional",kc),("unconditional",ku)):
            source=[c for c in counts if c["arm"]==arm and c["label"]==str(i)]
            assert len(source)==1 and int(source[0]["correct"])==x and int(source[0]["n"])==100
        ntest+=n;correct+=k;cond+=kc;uncond+=ku
    assert (ntest,correct,cond,uncond)==(270,257,778,94)
    assert m["train_n"]==1257 and m["validation_n"]==270 and m["test_n"]==270
    assert math.isclose(m["real_evaluator_accuracy"],257/270,abs_tol=1e-12)
    assert m["original_test_index_sha256"]=="d945a001641ccbaecf31384feb1d1ba6e54c4d0dc33a3f6c9a40effe505514bb"
    assert (m["recorded_conditional_agreement"],m["recorded_unconditional_comparison_agreement"])==(.778,.094)
    for arm,total in (("conditional",778),("unconditional",94)):
        overall=[c for c in counts if c["arm"]==arm and c["label"]=="overall"]
        assert len(overall)==1 and int(overall[0]["correct"])==total and int(overall[0]["n"])==1000
    print(json.dumps({"source":"original sklearn handwritten digits / prior actual generated samples",
      "independent_real_test_images":ntest,"correct_real":correct,
      "conditional_generated_label_agreement":cond,"unconditional_assigned_label_agreement":uncond,
      "result":"PASS","scope":"numerical source-file reconciliation only"},indent=2))
if __name__=="__main__":
    main()
