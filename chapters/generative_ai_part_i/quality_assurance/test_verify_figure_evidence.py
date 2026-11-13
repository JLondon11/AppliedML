"""Unit tests for Chapter 6 evidence checks (standard library only)."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from verify_figure_evidence import verify


class FigureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "example.py").write_text("print('example')\n")
        (self.root / "figure.png").write_bytes(b"image bytes")
        self.row = {
            "figure_id": "6.test", "manuscript_domain": "digits",
            "evidence_domain": "digits", "dataset": "real observations",
            "source_code": "example.py", "output": "figure.png",
            "output_sha256": hashlib.sha256(b"image bytes").hexdigest(),
            "experiment_executed": True, "dataset_provenance_verified": True,
            "manuscript_mapping_verified": True,
        }

    def tearDown(self):
        self.temp.cleanup()

    def check(self, **updates):
        return verify({"figures": [{**self.row, **updates}]}, self.root)

    def test_valid_metadata(self):
        self.assertEqual(self.check(), [])

    def test_domain_mismatch(self):
        self.assertTrue(any("domain mismatch" in x for x in self.check(evidence_domain="industrial")))

    def test_unverified_mapping(self):
        self.assertTrue(any("mapping not verified" in x for x in self.check(manuscript_mapping_verified=False)))

    def test_invalid_digest(self):
        self.assertTrue(any("64 lowercase hex" in x for x in self.check(output_sha256="UNVERIFIED")))

    def test_digest_mismatch(self):
        self.assertTrue(any("SHA-256 mismatch" in x for x in self.check(output_sha256="0" * 64)))

    def test_traversal(self):
        self.assertTrue(any("escapes repository root" in x for x in self.check(source_code="../example.py")))

    def test_execution_unverified(self):
        self.assertTrue(any("not attested" in x for x in self.check(experiment_executed=False)))

    def test_duplicate(self):
        self.assertTrue(any("duplicate figure" in x for x in verify({"figures": [self.row, self.row]}, self.root)))

    def test_empty(self):
        self.assertTrue(verify({"figures": []}, self.root))


if __name__ == "__main__":
    unittest.main()
