"""Regression check: Chapter 6 scoped certification verifier accepts exactly four records.

Run from anywhere:
    python -m unittest chapters/generative_ai_part_i/qa/test_scoped_certification_validator.py
This tests QA register accounting, not the scientific truth of unpublished artifacts.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class ScopedCertificationRegisterTests(unittest.TestCase):
    def test_four_scoped_certificates_and_production_hold(self):
        completed = subprocess.run(
            [sys.executable, str(HERE / "validate_scoped_ch6_certifications.py")],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode, 0,
            f"Verifier failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["scoped_scientific_certificates"], 4)
        self.assertEqual(result["levels"], {"A": 3, "B": 1})
        self.assertEqual(result["original_plant_heldouts"], 21)
        self.assertEqual(result["brownlee_90pct_coverage"], 19)
        self.assertEqual(result["chapter6_parent_applications_certified_by_this_register"], 0)
        self.assertEqual(result["production_status"], "HOLD")


if __name__ == "__main__":
    unittest.main()
