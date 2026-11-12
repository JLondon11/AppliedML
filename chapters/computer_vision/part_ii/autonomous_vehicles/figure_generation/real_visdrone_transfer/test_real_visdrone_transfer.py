"""Regression checks using original recorded VisDrone CSV, not simulated experiments.

Mutation cases corrupt copies of source summaries solely to test rejection;
none are experimental results.
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(script, classes, summary, out):
    return subprocess.run([sys.executable, str(script), "--per-class", str(classes),
                           "--summary", str(summary), "--output", str(out)],
                          capture_output=True, text=True)


def edit_summary(source, dest, name, value):
    with source.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    for row in records:
        if row["metric"] == name:
            row["value"] = str(value)
            break
    else:
        raise AssertionError(f"Missing metric {name}")
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True, type=Path)
    p.add_argument("--per-class", required=True, type=Path)
    p.add_argument("--summary", required=True, type=Path)
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="visdrone_evidence_") as d:
        d = Path(d)
        original = run(a.script, a.per_class, a.summary, d / "pass")
        if original.returncode:
            raise AssertionError("Original genuine evidence failed: " + original.stderr)
        meta = json.loads((d / "pass" / "source_metrics_verified.json").read_text())
        macro = meta.get("recomputed_macro_AP50", meta.get("macro_ap50"))
        recall = meta.get("recomputed_micro_recall", meta.get("micro_recall"))
        assert abs(macro - 0.1406456330251602) < 1e-12
        assert abs(recall - 0.33127572016460904) < 1e-12
        assert meta.get("n_gt", meta.get("gt_objects")) == 972
        assert meta.get("n_detections", meta.get("reported_detections")) == 2589
        undefined = meta.get("undefined_AP_excluded", meta.get("excluded_undefined_ap_classes"))
        assert undefined == ["bus"], undefined
        for metric, corrupt_value in (("local_macro_AP50", 0.9),
                                      ("local_micro_recall_at_cutoff", 0.9),
                                      ("n_original_gt", 1),
                                      ("n_detections", 1),
                                      ("class_count_evaluated", 1)):
            altered = d / f"bad_{metric}.csv"
            edit_summary(a.summary, altered, metric, corrupt_value)
            bad = run(a.script, a.per_class, altered, d / f"reject_{metric}")
            assert bad.returncode != 0, f"Pipeline accepted corrupt {metric}"
        print("PASS: original measured-data reconciliation, bus undefined-AP handling, five corrupt-summary rejections")
        print(f"AP50={macro:.12f}, recall={recall:.12f}, GT=972, detections=2589")


if __name__ == "__main__":
    main()
