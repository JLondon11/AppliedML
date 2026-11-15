"""Fixture-only regression tests for fail-closed production inventory coverage.

Run: python -m unittest discover -s chapters/generative_ai_part_i/qa
    -p 'test_ch6_production_inventory.py' -v
Passing synthetic software fixtures is NOT experimental evidence or manuscript QA.
"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_ch6_production_inventory import verify


class ProductionInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        base = self.root / "chapters/generative_ai_part_i"
        self.base = base
        (base / "quality_assurance").mkdir(parents=True)
        (base / "manuscript").mkdir()
        pdf = base / "manuscript/ch6.pdf"
        pdf.write_bytes(b"%PDF-1.7 test-fixture only")
        generator = base / "code/generate.py"
        generator.parent.mkdir()
        generator.write_text("# test fixture; not a measured experiment\n")
        results = base / "code/results.csv"
        results.write_text("a\n0\n")
        figures = []
        for n in range(1, 49):
            image = base / f"figures/fig{n:02d}.svg"
            image.parent.mkdir(exist_ok=True)
            image.write_text(f"<svg><!-- fixture {n} --></svg>")
            figures.append({
                "figure_id": f"6.{n:02d}",
                "asset": str(image.relative_to(self.root)),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "caption": f"Test fixture {n}",
                "scientific_reviewed": True,
                "kind": "experimental" if n <= 14 or n == 20 else "conceptual",
                **({"dataset_provenance_verified": True,
                    "generator": str(generator.relative_to(self.root)),
                    "observed_data_or_results": str(results.relative_to(self.root))}
                   if n <= 14 or n == 20 else {}),
            })
        self.inventory = {
            "chapter": 6,
            "compiled_source_reviewed": True,
            "manuscript_pdf": str(pdf.relative_to(self.root)),
            "manuscript_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "figures": figures,
            "parent_applications": [
                {"unit_id": f"fixture-parent-{n:02d}", "exact_manuscript_title": f"Fixture {n}",
                 "scope_reviewed": True, "figure_ids": [f"6.{n:02d}"],
                 "scientific_status": "SCIENTIFIC_HOLD"} for n in range(1, 15)
            ],
            "tables": [
                {"table_id": f"6.T{n:02d}", "substantive_reviewed": True,
                 "source_or_derivation": "test-only fixture, not experimental evidence"}
                for n in range(1, 18)
            ],
            "code_listings": [
                {"listing_id": f"6.L{n:02d}", "code_reviewed": True,
                 "complete_code": str(generator.relative_to(self.root))}
                for n in range(1, 12)
            ],
        }
        self.path = base / "quality_assurance/production_inventory.json"

    def run_inventory(self, data=None):
        self.path.write_text(json.dumps(self.inventory if data is None else data))
        return verify(self.root)

    def test_complete_software_fixture_passes_structural_only(self):
        result = self.run_inventory()
        self.assertEqual(result["status"], "STRUCTURAL_PASS_MANUAL_REVIEW_REQUIRED", result)
        self.assertEqual(result["issues"], [])

    def test_missing_inventory_fails_closed(self):
        result = verify(self.root)
        self.assertEqual(result["status"], "BLOCKED")

    def test_one_figure_cannot_pass(self):
        data = copy.deepcopy(self.inventory)
        data["figures"] = data["figures"][:1]
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_missing_parent_application_cannot_pass(self):
        data = copy.deepcopy(self.inventory)
        data["parent_applications"].pop()
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_tampered_artifact_fails(self):
        data = copy.deepcopy(self.inventory)
        data["figures"][19]["sha256"] = "0" * 64
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_untraced_experiment_fails(self):
        data = copy.deepcopy(self.inventory)
        del data["figures"][19]["generator"]
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_parent_without_empirical_figure_fails(self):
        data = copy.deepcopy(self.inventory)
        data["parent_applications"][0]["figure_ids"] = ["6.48"]
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_unverified_dataset_source_fails(self):
        data = copy.deepcopy(self.inventory)
        data["figures"][19]["dataset_provenance_verified"] = False
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_unsupported_experiment_promotion_fails(self):
        data = copy.deepcopy(self.inventory)
        data["parent_applications"][0]["scientific_status"] = "SCIENTIFIC_CERTIFIED"
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_missing_code_listing_fails(self):
        data = copy.deepcopy(self.inventory)
        data["code_listings"].pop()
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_missing_table_fails(self):
        data = copy.deepcopy(self.inventory)
        data["tables"].pop()
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_missing_listing_source_fails(self):
        data = copy.deepcopy(self.inventory)
        data["code_listings"][0]["complete_code"] = "missing.py"
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")

    def test_unjustified_table_fails(self):
        data = copy.deepcopy(self.inventory)
        data["tables"][0]["substantive_reviewed"] = False
        self.assertEqual(self.run_inventory(data)["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
