"""Software regression tests for genuine measured NEU paired training-seed analysis."""
import csv
import tempfile
import unittest
from pathlib import Path

from audit_paired_seed_uncertainty import analyze, draw_svg

ROOT = Path(__file__).resolve().parent
MEASURED = ROOT / "replicated_seed_metrics_pass13.csv"

class OriginalNEUPairedTests(unittest.TestCase):
    def test_actual_source_effects_and_sign_tail(self):
        result,rows=analyze(MEASURED)
        self.assertEqual(len(rows),12)
        self.assertEqual(result["source_training_seeds"],[20261106,20261107,20261108])
        for metric,baseline,expected in (
            ("test_accuracy","real_only",-0.19259259263333336),
            ("test_macro_f1","real_only",-0.2061735354333333),
            ("test_accuracy","real_repeat_augmented",-0.2694444444666667),
            ("test_macro_f1","real_repeat_augmented",-0.2870857760666667),
        ):
            record=result["metrics"][metric][baseline]
            self.assertAlmostEqual(record["mean_difference"],expected,places=10)
            self.assertEqual(record["negative_difference_seeds"],3)
            self.assertEqual(record["one_sided_exact_sign_test_tail"],.125)
            self.assertEqual(len(record["three_paired_seed_differences"]),3)
        self.assertIn("three training seeds",draw_svg(result).lower())

    def edit(self,action):
        with MEASURED.open(newline="",encoding="utf-8") as fh:
            records=list(csv.DictReader(fh))
        fields=list(records[0])
        action(records)
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"corrupted.csv"
            with file.open("w",newline="",encoding="utf-8") as fh:
                writer=csv.DictWriter(fh,fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            with self.assertRaises(ValueError):
                analyze(file)

    def test_missing_real_arm_fails(self):
        self.edit(lambda rows: rows.pop())

    def test_duplicate_original_seed_arm_fails(self):
        self.edit(lambda rows: rows.__setitem__(0, dict(rows[1])))

    def test_unregistered_seed_fails(self):
        self.edit(lambda rows: rows[0].__setitem__("seed","20261109"))

    def test_impossible_accuracy_fails(self):
        self.edit(lambda rows: rows[0].__setitem__("test_accuracy","1.01"))

    def test_nonfinite_macro_f1_fails(self):
        self.edit(lambda rows: rows[0].__setitem__("test_macro_f1","nan"))

    def test_numerical_schema_error_fails(self):
        self.edit(lambda rows: rows[0].__setitem__("test_accuracy","not-a-measurement"))

if __name__=="__main__":
    unittest.main()
