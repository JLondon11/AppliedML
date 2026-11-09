"""Regression tests for canonical Chapter 9 evidence gate."""
import base64
import hashlib
import json
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chapters/generative_ai_part_ii/qa"))
from audit_figure_evidence import audit, frozen_specs


class TestFigureEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.figure = self.root / "figure.png"
        self.metrics = self.root / "metrics.json"
        self.figure.write_bytes(b"measured figure bytes")
        self.metrics.write_text('{"measured": true}')
        self.specs = [[1, "9.1", "figure_01.png", "REAL DATA", "HOLD", 7, "demo", "hash"]]
        self.record = {
            "figure_number": 1, "dataset": "named real dataset",
            "generator": "chapters/generative_ai_part_ii/code/train.py",
            "experiment_protocol": "disjoint train validation and test; trained model",
            "figure_path": "figure.png", "metrics_path": "metrics.json",
            "figure_sha256": hashlib.sha256(self.figure.read_bytes()).hexdigest(),
            "metrics_sha256": hashlib.sha256(self.metrics.read_bytes()).hexdigest(),
            "model_or_method": "trained experimental model", "status": "VERIFIED"}

    def test_missing_record_is_hold(self):
        self.assertEqual(audit(self.specs, {"figures": []}, self.root)[0]["status"], "HOLD")

    def test_exact_hashes_verify(self):
        self.assertEqual(audit(self.specs, {"figures": [self.record]}, self.root)[0]["status"], "VERIFIED")

    def test_corrupt_figure_is_hold(self):
        self.figure.write_bytes(b"changed")
        self.assertEqual(audit(self.specs, {"figures": [self.record]}, self.root)[0]["status"], "HOLD")

    def test_fallback_generator_is_hold(self):
        self.record["generator"] = "chapters/_shared/figure_generation_runtime.py"
        self.assertEqual(audit(self.specs, {"figures": [self.record]}, self.root)[0]["status"], "HOLD")

    def test_duplicate_evidence_rejected(self):
        with self.assertRaises(ValueError):
            audit(self.specs, {"figures": [self.record, self.record]}, self.root)

    def test_decode_frozen_spec(self):
        blob = base64.b64encode(zlib.compress(json.dumps(self.specs).encode())).decode()
        f = self.root / "generator.py"
        f.write_text("SPEC_BLOB=" + repr(blob))
        self.assertEqual(frozen_specs(f), self.specs)


if __name__ == "__main__":
    unittest.main()
