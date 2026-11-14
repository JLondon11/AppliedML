"""Independent record-level audit for original Yeh concrete held-out observations.

Run from any directory:
python chapters/generative_ai_part_i/qa/audit_concrete_retrospective.py

Uses only Python standard library and already committed files. Does not train a
model, independently verify physical properties of generated mixes, or certify
the entire concrete design application.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "code" / "section_6_engineering_concrete_mix_generation"
EPS = 1e-9


def records(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(a, b, tol=EPS):
    return math.isclose(float(a), float(b), abs_tol=tol, rel_tol=tol)


def audit():
    original = records(DATA / "Concrete_Data_Yeh.csv")
    held = records(DATA / "heldout_test_predictions.csv")
    design = records(DATA / "conditional_generated_designs.csv")
    summary = json.loads((DATA / "benchmark_summary.json").read_text(encoding="utf-8"))
    if len(original) != 1030 or len(held) != 229 or len(design) != 200:
        raise AssertionError("Unexpected authentic row, held-out, or saved-design count")
    if summary["split_rows"] != {"train": 645, "validation": 156, "test": 229}:
        raise AssertionError("Unexpected original split summary")
    ids = [int(row["id"]) for row in held]
    if len(ids) != len(set(ids)) or any(i < 1 or i > len(original) for i in ids):
        raise AssertionError("Held-out row identifier invalid or repeated")
    headers = ("cement", "slag", "flyash", "water", "superplasticizer",
               "coarseaggregate", "fineaggregate")
    def group(row):
        return tuple(float(row[c]) for c in headers)
    held_set = set(ids)
    train_val_groups = {group(row) for i, row in enumerate(original, 1) if i not in held_set}
    held_groups = {group(original[i-1]) for i in ids}
    if held_groups & train_val_groups:
        raise AssertionError("Identical mix formula leaked across test and development partitions")
    sq = ab = 0.
    day28 = []
    for row in held:
        src = original[int(row["id"])-1]
        for col in ("cement", "water", "age"):
            if not close(row[col], src[col]):
                raise AssertionError(f"Original source mismatch at {row['id']} / {col}")
        if not close(row["strength_measured"], src["csMPa"]):
            raise AssertionError(f"Measured source strength mismatch at {row['id']}")
        binder = sum(float(src[c]) for c in ("cement", "slag", "flyash"))
        if not close(row["binder"], binder):
            raise AssertionError(f"Binder mismatch at {row['id']}")
        error = float(row["strength_predicted"]) - float(src["csMPa"])
        sq += error * error
        ab += abs(error)
        if close(src["age"], 28):
            day28.append(row)
    rmse = math.sqrt(sq / len(held))
    mae = ab / len(held)
    if not close(rmse, summary["test_knn"]["rmse"], 1e-8):
        raise AssertionError("Held-out RMSE does not reproduce recorded summary")
    if not close(mae, summary["test_knn"]["mae"], 1e-8):
        raise AssertionError("Held-out MAE does not reproduce recorded summary")
    eligible = [r for r in day28 if float(r["cement"]) <= summary["day28"]["cement_median_train"]
                and float(r["water"]) / float(r["binder"]) <= 0.65]
    if (len(day28), len(eligible)) != (86, 29):
        raise AssertionError("28-day observed-data screen count mismatch")
    hits = sum(float(r["strength_measured"]) >= 40 for r in eligible)
    if hits != 0 or hits != summary["day28"]["observed_high_strength_eligible"]:
        raise AssertionError("Measured 40 MPa hit count mismatch")
    arms = {}
    for row in design:
        arm = row["arm"]
        arms[arm] = arms.get(arm, 0) + 1
        if row["source"] != "model generated, not laboratory tested":
            raise AssertionError("Generated candidate must not be labeled observed")
        if not math.isfinite(float(row["predicted_strength"])):
            raise AssertionError("Non-finite surrogate strength")
        if int(row["physical_screen_pass"]) not in (0, 1):
            raise AssertionError("Invalid rule-screen flag")
    expected = {"high_strength_conditioned": 100, "unconditioned_training_mixes": 100}
    if arms != expected:
        raise AssertionError("Saved generator arm counts inconsistent")
    # Git blob ID is a source-byte identity, NOT a second independent archive.
    source_bytes = (DATA / "Concrete_Data_Yeh.csv").read_bytes()
    blob_sha = hashlib.sha1(b"blob " + str(len(source_bytes)).encode()
                            + b"\x00" + source_bytes).hexdigest()
    if blob_sha != summary["source_gh_blob_sha"]:
        raise AssertionError("Original source blob hash mismatch")
    return {"record_level_check": "PASS", "source_rows": 1030,
            "heldout_observations": 229, "unique_test_groups": len(held_groups),
            "heldout_rmse_mpa": rmse, "heldout_mae_mpa": mae,
            "real_28day_observations": len(day28),
            "real_screen_eligible": len(eligible),
            "real_screen_measured_40mpa_hits": hits,
            "saved_generated_proposals": arms,
            "independent_physical_mix_tests": 0,
            "status": "RETROSPECTIVE_DATA_AUDIT_ONLY"}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
