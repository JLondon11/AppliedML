"""Exact original-assay hypergeometric reference for retrospective ESOL top-20 ranking.

Input: saved original measured held-out records; no external data needed.
Distinguishes real measured labels from precomputed published model scores,
and does not claim prospective/new-molecule screening or fair retraining.
"""
import argparse
import csv
import json
import math
from pathlib import Path

def evaluate(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    if len(records) != 226 or len({r["smiles"] for r in records}) != 226:
        raise ValueError("Expected 226 unique original held-out compounds")
    for row in records:
        y = float(row["measured_logS"])
        if not math.isfinite(y) or any(not math.isfinite(float(row[k])) for k in ("ridge", "published_ESOL")):
            raise ValueError("Nonfinite source value")
        if int(row["hit"]) != int(y >= -1.6):
            raise ValueError("Original hit and training-only threshold mismatch")
    N, n = 226, 20
    K = sum(int(r["hit"]) for r in records)
    if K != 59:
        raise ValueError("Unexpected number of genuine high-solubility original assays")
    den = math.comb(N, n)
    pmf = [math.comb(K,k)*math.comb(N-K,n-k)/den for k in range(n+1)]
    if not math.isclose(sum(pmf), 1.0, abs_tol=1e-12):
        raise ValueError("Hypergeometric mass not normalized")
    def quantile(q):
        cdf = 0.
        for k,p in enumerate(pmf):
            cdf += p
            if cdf >= q: return k
        raise ValueError("Invalid finite-population distribution")
    models = {}
    for model in ("ridge", "published_ESOL"):
        top = sorted(records, key=lambda r: float(r[model]), reverse=True)[:n]
        hits = sum(int(r["hit"]) for r in top)
        rmse = math.sqrt(sum((float(r[model])-float(r["measured_logS"]))**2
                             for r in records)/N)
        models[model] = {
            "measured_hits_top20":hits,
            "precision_top20":hits/n,
            "fold_enrichment_over_random":(hits/n)/(K/N),
            "exact_uniform_random_upper_tail_probability":sum(pmf[hits:]),
            "measured_test_rmse_log_mol_L":rmse,
        }
    if (models["ridge"]["measured_hits_top20"],
        models["published_ESOL"]["measured_hits_top20"]) != (14,17):
        raise ValueError("Ranked source assay hit counts do not reconcile")
    return {
        "source":"verified_original_ESOL_226_test_predictions.csv",
        "n_original_heldout_assays":N,"observed_high_solubility_assays":K,
        "train_derived_high_solubility_threshold_logS":-1.6,
        "random_selection_n":n,"random_selection_expected_hits":n*K/N,
        "random_selection_central_95pct_hits":[quantile(.025),quantile(.975)],
        "model_arms":models,
        "limitations":[
            "Exact tail probabilities refer to a uniform random top-20 selection without replacement from these fixed 226 observed assays",
            "An enrichment p-value does not certify prospective screening of unassayed generated molecules",
            "Only one preselected train/test split and one fitted local ridge model are evaluated",
            "The published ESOL score is a precomputed input column with unverified original training independence, not a fair independently trained baseline",
            "No physical synthesis, prospective solubility assay or autonomous discovery occurred"
        ]
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = json.dumps(evaluate(args.csv), indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result,encoding="utf-8")
    print(result)
