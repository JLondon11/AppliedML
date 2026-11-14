"""Regression tests for committed Chapter 6 evidence-manifest schemas.

Run: python -m unittest discover -s chapters/generative_ai_part_i/figure_generation \
    -p 'test_verify_source_provenance.py'
Tests deterministic integrity fixtures, not the chapter's scientific conclusions.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from verify_source_provenance import digest, git_blob_sha, inspect


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "vae_pca_real_digits.py"
        self.source.write_text("print('fixture')\n", encoding="utf-8")
        self.output = self.root / "output"
        self.output.mkdir()
        (self.output / "metrics.json").write_text('{"fixture": 1}\n')
        self.blob = git_blob_sha(self.source)

    def manifest(self):
        return {
            "source_git_blob_sha1": self.blob,
            "files_sha256": {
                self.source.name: digest(self.source),
                "output/metrics.json": digest(self.output / "metrics.json"),
            },
        }

    def test_actual_package_format_passes(self):
        result = inspect(self.source, self.blob, self.root, self.manifest())
        self.assertEqual(result["failed"], 0, result["checks"])

    def test_legacy_files_format_passes(self):
        result = inspect(self.source, self.blob, self.output,
                         {"files": {"metrics.json": digest(self.output / "metrics.json")}})
        self.assertEqual(result["failed"], 0, result["checks"])

    def test_modified_source_fails(self):
        original = self.manifest()
        self.source.write_text("print('changed fixture')\n", encoding="utf-8")
        self.assertGreater(inspect(self.source, self.blob, self.root, original)["failed"], 0)

    def test_modified_output_fails(self):
        original = self.manifest()
        (self.output / "metrics.json").write_text('{"fixture": 2}\n')
        self.assertGreater(inspect(self.source, self.blob, self.root, original)["failed"], 0)

    def test_unsafe_traversal_is_rejected(self):
        manifest = self.manifest()
        manifest["files_sha256"]["../outside.txt"] = "a" * 64
        self.assertGreater(inspect(self.source, self.blob, self.root, manifest)["failed"], 0)

    def test_missing_source_checksum_is_rejected(self):
        manifest = self.manifest()
        del manifest["files_sha256"][self.source.name]
        self.assertGreater(inspect(self.source, self.blob, self.root, manifest)["failed"], 0)

    def test_conflicting_manifest_formats_fail(self):
        manifest = self.manifest()
        manifest["files"] = {"output/metrics.json": digest(self.output / "metrics.json")}
        self.assertGreater(inspect(self.source, self.blob, self.root, manifest)["failed"], 0)

    def test_manifest_blob_disagreement_fails(self):
        manifest = self.manifest()
        manifest["source_git_blob_sha1"] = "0" * 40
        self.assertGreater(inspect(self.source, self.blob, self.root, manifest)["failed"], 0)


if __name__ == "__main__":
    unittest.main()
