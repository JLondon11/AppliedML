"""Unit tests for fail-closed Chapter 6 production readiness checks.

These tests verify gate behavior on controlled fixtures, not scientific validity
of any Chapter 6 model. Run: python -m unittest discover -s
chapters/generative_ai_part_i/qa -p 'test_ch6_release_gate.py'
"""
from __future__ import annotations
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from check_ch6_release_gate import check


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.chapter = self.root / "chapters/generative_ai_part_i"
        (self.chapter / "qa").mkdir(parents=True)
        (self.chapter / "quality_assurance").mkdir()
        self.reg = self.chapter / "qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv"
        fields = ["unit_id", "level", "scientific_status", "production_status", "evidence_limits"]
        fields += [f"R{i}" for i in range(1, 13)]
        with self.reg.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for n in range(1, 11):
                writer.writerow({
                    "unit_id": f"CH6-D{n:02d}", "level": "B" if n == 4 else "A",
                    "scientific_status": "SCIENTIFIC_CERTIFIED",
                    "production_status": "PRODUCTION_HOLD",
                    "evidence_limits": "fixture only; not a scientific certificate",
                    **{f"R{i}": "PASS" for i in range(1, 11)}})

    def _manifest(self, *, mismatched=False, with_caption=True):
        src = Path("chapters/generative_ai_part_i/figure_generation/industrial_experiment.py")
        out = Path("chapters/generative_ai_part_i/figures/industrial620.png")
        (self.root / src).parent.mkdir(parents=True, exist_ok=True)
        (self.root / src).write_text("# fixture only\n")
        (self.root / out).parent.mkdir(parents=True, exist_ok=True)
        (self.root / out).write_bytes(b"test-only-fixture; not an image")
        entry = {
            "figure_id": "6.20", "manuscript_domain": "industrial-defect-images",
            "evidence_domain": "handwritten-digits" if mismatched else "industrial-defect-images",
            "source_code": str(src), "output": str(out),
            "output_sha256": hashlib.sha256((self.root / out).read_bytes()).hexdigest(),
            "manuscript_caption": "Industrial defect, test-only fixture" if with_caption else "",
            "manuscript_mapping_verified": True, "experiment_executed": True,
            "dataset_provenance_verified": True,
        }
        path = self.chapter / "quality_assurance/chapter6_evidence_manifest.json"
        path.write_text(json.dumps({"figures": [entry]}))

    def _manuscript(self):
        d = self.chapter / "manuscript"
        d.mkdir()
        (d / "chapter6.tex").write_text("\\documentclass{book}\n\\begin{document}\nFixture\\end{document}\n")
        (d / "chapter6.pdf").write_bytes(b"%PDF-1.7 fixture only")

    def test_missing_scoped_certificate_blocks_release(self):
        # Removing a real scoped certificate must never silently reduce the gate.
        with self.reg.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        with self.reg.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[:-1])
        self._manifest()
        self._manuscript()
        report = check(self.root)
        self.assertEqual(report["release"], "BLOCKED")
        self.assertIn("scoped_id_continuity",
                      {r["gate"] for r in report["checks"] if not r["pass"]})

    def test_noncontiguous_scoped_id_blocks_release(self):
        with self.reg.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[-1]["unit_id"] = "CH6-D12"
        with self.reg.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self._manifest()
        self._manuscript()
        report = check(self.root)
        self.assertEqual(report["release"], "BLOCKED")
        self.assertIn("scoped_id_continuity",
                      {r["gate"] for r in report["checks"] if not r["pass"]})

    def test_missing_artifacts_fail_closed(self):
        result = check(self.root)
        self.assertEqual(result["release"], "BLOCKED")
        self.assertGreater(result["failed"], 0)

    def test_scientific_scopes_do_not_grant_production_release(self):
        self._manifest()
        result = check(self.root)
        self.assertEqual(result["release"], "BLOCKED")
        blocked = {v["gate"] for v in result["checks"] if not v["pass"]}
        self.assertIn("current_manuscript_source", blocked)
        self.assertIn("current_manuscript_pdf", blocked)

    def test_mismatched_figure_620_is_blocked(self):
        self._manifest(mismatched=True)
        self._manuscript()
        result = check(self.root)
        self.assertEqual(result["release"], "BLOCKED")
        self.assertIn("figure_620_mapping", {x["gate"] for x in result["checks"] if not x["pass"]})

    def test_missing_caption_is_blocked(self):
        self._manifest(with_caption=False)
        self._manuscript()
        self.assertEqual(check(self.root)["release"], "BLOCKED")

    def test_single_figure_is_not_full_production_coverage(self):
        self._manifest()
        self._manuscript()
        result = check(self.root)
        self.assertEqual(result["release"], "BLOCKED")
        self.assertIn("full_production_inventory",
                      {r["gate"] for r in result["checks"] if not r["pass"]})


if __name__ == "__main__":
    unittest.main()
