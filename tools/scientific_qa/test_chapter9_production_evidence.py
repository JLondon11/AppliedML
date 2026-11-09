"""Production-gate regression tests: empirical evidence is required, never assumed."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
    "chapters/generative_ai_part_ii/qa"))
from production_evidence import validate_production


class ProductionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.base=self.root/"chapters/generative_ai_part_ii"
        (self.base/"code").mkdir(parents=True)
        (self.base/"figures").mkdir()
        self.gen=self.base/"code/experiment.py"
        self.figure=self.base/"figures/output.png"
        self.metrics=self.base/"code/measured.json"
        self.gen.write_text("print('actual experiment')\n")
        self.figure.write_bytes(bytes.fromhex("89504e470d0a1a0a")+b"fixture")
        self.metrics.write_text(json.dumps({"test_accuracy":0.91}))
        self.r={
           "status":"VERIFIED",
           "evidence_type":"measured_real_dataset",
           "figure_number":1,
           "dataset_name":"Example real dataset",
           "dataset_source_url":"https://example.org/dataset",
           "split_counts":{"train":70,"validation":15,"test":15},
           "split_disjoint":True,
           "caption":"Held-out measurements of a trained model on a named real dataset, with explicit partitions and evaluation limitations.",
           "method":"trained experimental classifier",
           "generator":"chapters/generative_ai_part_ii/code/experiment.py",
           "generator_sha256":hashlib.sha256(self.gen.read_bytes()).hexdigest(),
           "figure_path":"chapters/generative_ai_part_ii/figures/output.png",
           "figure_sha256":hashlib.sha256(self.figure.read_bytes()).hexdigest(),
           "metrics_path":"chapters/generative_ai_part_ii/code/measured.json",
           "metrics_sha256":hashlib.sha256(self.metrics.read_bytes()).hexdigest(),
        }

    def check(self):
        return validate_production(self.r,self.root,1)

    def test_valid_structural_evidence(self):
        self.assertEqual(self.check(),[])

    def test_bad_dataset_split_fails(self):
        self.r["split_counts"]["test"]=0
        self.assertTrue(any("positive disjoint" in x for x in self.check()))

    def test_missing_real_evidence_fails(self):
        self.r["evidence_type"]="synthetic_simulation"
        self.assertTrue(any("measured_real_dataset" in x for x in self.check()))

    def test_modified_figure_fails(self):
        self.figure.write_bytes(b"changed")
        self.assertTrue(any("PNG header" in x for x in self.check()))
        self.assertTrue(any("SHA256 mismatch" in x for x in self.check()))

    def test_empty_metrics_fails(self):
        self.metrics.write_text('{"note": "no measurements"}')
        self.r["metrics_sha256"]=hashlib.sha256(self.metrics.read_bytes()).hexdigest()
        self.assertTrue(any("no measured" in x for x in self.check()))

    def test_wrong_chapter_source_fails(self):
        self.r["generator"]="chapters/_shared/figure_generation_runtime.py"
        self.assertTrue(any("generic fallback" in x for x in self.check()))

    def test_path_escape_fails(self):
        self.r["figure_path"]="../../outside.png"
        self.assertTrue(any("escapes repository" in x for x in self.check()))


if __name__=="__main__":
    unittest.main()
