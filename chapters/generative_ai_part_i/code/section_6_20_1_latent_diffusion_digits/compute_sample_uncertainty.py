"""Calculate descriptive Wilson intervals for Chapter 6 digit evaluator evidence.

These are conditional sample-level intervals, NOT model-training-seed uncertainty,
independent generator quality estimates, or two-sample significance tests.

python compute_sample_uncertainty.py
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "original_evaluator_control" / "real_evaluator_vs_generated_class.csv"
OUT = ROOT / "original_evaluator_control" / "sample_uncertainty.json"
Z = 1.959963984540054

def wilson(k, n):
    p = k / n
    z2 = Z * Z
    center = (p + z2 / (2*n)) / (1 + z2/n)
    half = Z * math.sqrt(p * (1-p)/n + z2/(4*n*n)) / (1+z2/n)
    return [center-half, center+half]

def main():
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    out_rows = []
    for row in rows:
        digit = int(row["digit"])
        real_k, real_n = int(row["real_test_correct"]), int(row["real_test_n"])
        generated_k, generated_n = int(row["conditional_generated_agree"]), int(row["conditional_generated_n"])
        assert real_n > 0 and generated_n == 100 and 0 <= real_k <= real_n and 0 <= generated_k <= generated_n
        out_rows.append({"digit":digit,"real_holdout_wilson95":wilson(real_k,real_n),
                         "saved_generated_agreement_wilson95_conditional_on_fitted_model":wilson(generated_k,generated_n)})
    assert [r["digit"] for r in out_rows] == list(range(10))
    real_k = sum(int(r["real_test_correct"]) for r in rows)
    real_n = sum(int(r["real_test_n"]) for r in rows)
    gen_k = sum(int(r["conditional_generated_agree"]) for r in rows)
    gen_n = sum(int(r["conditional_generated_n"]) for r in rows)
    assert (real_k,real_n,gen_k,gen_n) == (257,270,778,1000)
    result = {"source":"real_evaluator_vs_generated_class.csv",
              "real_test":{"correct":real_k,"n":real_n,"wilson95":wilson(real_k,real_n)},
              "conditional_generated":{"classifier_agree":gen_k,"n":gen_n,
                  "wilson95_conditional_on_fitted_generator":wilson(gen_k,gen_n)},
              "per_digit":out_rows,
              "limitations":["Real observed and generated image groups are distinct and are not paired",
                  "Sample Wilson intervals do not quantify generator-training-seed uncertainty or image dependence",
                  "Automated classifier agreement is not perceptual image fidelity",
                  "No hypothesis test of a common population was performed"]}
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"real_test_ci95":result["real_test"]["wilson95"],
                      "generated_ci95":result["conditional_generated"]["wilson95_conditional_on_fitted_generator"]},indent=2))

if __name__ == "__main__":
    main()
