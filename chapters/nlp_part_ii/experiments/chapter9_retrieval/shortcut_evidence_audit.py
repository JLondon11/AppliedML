#!/usr/bin/env python3
"""Post-hoc audit of saved Chapter 9 retrieval predictions.

Reads frozen manuscript text, fixed questions, saved rankings and extractive
evidence traces supplied in the companion execution package. It neither
trains a model nor certifies any of Chapter 9's eight applications.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

def read_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def terms(value):
    stop = set("what why how does are and the for with from into between which "
               "when where these that this they their have more than using used "
               "make about should can could model models system systems method "
               "methods section role do in at of by on to as a an is its it".split())
    return {token for token in re.findall(r"[a-z][a-z0-9]+", value.lower())
            if len(token) >= 3 and token not in stop}

def wilson(successes, count, z=1.95996398454):
    p = successes / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    half = z * np.sqrt((p * (1-p) + z*z/(4*count)) / count) / scale
    return [round(float(center-half), 4), round(float(center+half), 4)]

def main():
    raw_file = ROOT / "source/chapter9_extracted_text.txt"
    raw = raw_file.read_text(encoding="utf-8")
    titles = {m.group(1):m.group(2).strip() for m in re.finditer(
        r"(?m)^(9\.\d+(?:\.\d+)?)\s{2,}([^\n]{5,110})\s*$", raw)}
    questions = read_csv("fixed_queries.csv")
    predictions = read_csv("sensitivity_query_results.csv")
    traces = read_csv("extractive_evidence_traces.csv")
    assert (len(questions), len(predictions), len(traces)) == (39, 39, 117)
    methods = ["BM25 (saved)", "Word TF-IDF (saved)",
               "Word+character TF-IDF (saved)", "Title-only TF-IDF"]
    rows = []
    for question, prediction in zip(questions, predictions):
        assert question["query"] == prediction["query"]
        assert question["target_section"] == prediction["gold_section"]
        title = titles[question["target_section"]]
        qterms, tterms = terms(question["query"]), terms(title)
        overlap = len(qterms & tterms) / len(tterms) if tterms else 0
        row = {"query": question["query"], "gold_section": question["target_section"],
               "title": title, "heading_term_overlap": round(overlap, 4)}
        row.update({method: int(int(prediction[method]) == 1) for method in methods})
        rows.append(row)
    groups = {"all": rows,
              "zero_title_term_overlap": [r for r in rows if r["heading_term_overlap"] == 0],
              "some_title_term_overlap": [r for r in rows if r["heading_term_overlap"] > 0],
              "majority_title_term_overlap": [r for r in rows if r["heading_term_overlap"] >= 0.5]}
    output = []
    for subset, samples in groups.items():
        for method in methods:
            n = len(samples)
            correct = sum(sample[method] for sample in samples)
            output.append({"subset": subset, "method": method,
                           "n": n, "correct": correct,
                           "accuracy": round(correct/n, 4),
                           "wilson95": wilson(correct, n)})
    flagged = []
    for trace in traces:
        sentence = trace["evidence_excerpt"]
        reasons = []
        if float(trace["evidence_tfidf_similarity"]) < 0.1:
            reasons.append("very_low_lexical_match")
        if re.search(r"[\ufffd\uf0b7]|\s[√∑∏≠≤≥]\s", sentence):
            reasons.append("possible_pdf_glyph_artifact")
        if re.search(r"\b[a-zA-Z]{3,}-\s+[a-zA-Z]{3,}\b", sentence):
            reasons.append("possible_pdf_hyphenation")
        if len(sentence.split()) < 10:
            reasons.append("short_excerpt")
        if reasons:
            flagged.append({"method": trace["method"], "query": trace["query"],
                            "retrieved_section": trace["retrieved_section"],
                            "flags": reasons, "excerpt": sentence})
    report = {"source_sha256": hashlib.sha256(raw_file.read_bytes()).hexdigest(),
              "n_questions": 39, "n_traces": 117, "n_flagged_traces": len(flagged),
              "method_subsets": output, "flagged_traces": flagged,
              "limitations": ["Post-hoc evaluation on author-authored questions.",
                              "Heading overlap is not proof of leakage.",
                              "PDF extraction may corrupt text or formula glyphs.",
                              "Provenance does not imply answer correctness.",
                              "No trained model was run and no application was certified."]}
    (ROOT / "shortcut_audit_results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"subsets": {k: len(v) for k,v in groups.items()},
                      "flagged": len(flagged), "original_counts":
                      {m:next(r["correct"] for r in output if r["subset"]=="all"
                              and r["method"]==m) for m in methods}}, indent=2))

if __name__ == "__main__":
    main()
