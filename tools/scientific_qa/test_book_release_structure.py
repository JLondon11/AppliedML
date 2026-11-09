"""Unit tests for the conservative book structural readiness checker."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_book_release_structure import CHAPTERS, audit, main

class ReleaseStructureTests(unittest.TestCase):
    def test_empty_tree_fails_all_twelve(self):
        with tempfile.TemporaryDirectory() as d:
            report = audit(Path(d))
            self.assertEqual(report["total_chapters"], 12)
            self.assertEqual(report["structurally_ready_chapters"], 0)
            self.assertFalse(report["book_structurally_ready"])
            self.assertFalse(report["book_scientifically_certified"])

    def test_complete_structure_does_not_certify_science(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for _, _, location in CHAPTERS:
                chapter = root / location
                for section, name in (
                    ("manuscript", "chapter.tex"),
                    ("manuscript", "chapter.pdf"),
                    ("figures", "figure.png"),
                    ("code", "example.py"),
                    ("instructor_solutions", "solutions.tex"),
                    ("qa", "review.md"),
                ):
                    target = chapter / section / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("content", encoding="utf-8")
                (chapter / "RELEASE_MANIFEST.md").write_text("manifest", encoding="utf-8")
            report = audit(root)
            self.assertEqual(report["structurally_ready_chapters"], 12)
            self.assertTrue(report["book_structurally_ready"])
            self.assertFalse(report["book_scientifically_certified"])
            self.assertTrue(all(not c["scientifically_certified"] for c in report["chapters"]))
            self.assertEqual(main(["--repo-root", d]), 0)

    def test_empty_placeholder_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / CHAPTERS[0][2] / "manuscript" / "chapter.tex"
            path.parent.mkdir(parents=True)
            path.touch()
            self.assertFalse(audit(Path(d))["chapters"][0]["checks"]["manuscript_latex"])

if __name__ == "__main__":
    unittest.main()
