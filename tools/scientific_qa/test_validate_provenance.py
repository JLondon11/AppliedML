"""Regression tests for the scientific evidence gate."""
import json
from pathlib import Path
import tempfile
import unittest

from validate_provenance import validate_manifest, validate_record


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(figure='01', category='empirical', status='unverified',
                           generator='run', dataset='named dataset',
                           protocol='documented split', artifact='outputs/01.png')

    def test_valid_unverified(self):
        self.assertEqual(validate_record(self.record, 0), [])

    def test_certification_requires_evidence(self):
        record = dict(self.record, status='certified')
        self.assertTrue(any('metrics_file' in x for x in validate_record(record, 0)))
        record.update(metrics_file='metrics.csv', test_command='python -m unittest',
                      test_result='passed')
        self.assertEqual(validate_record(record, 0), [])

    def test_illustration_cannot_be_certified(self):
        record = dict(self.record, category='illustration', status='certified')
        self.assertTrue(any('conceptual' in x for x in validate_record(record, 0)))

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps({'figures': [self.record, self.record]}))
            self.assertTrue(any('duplicate' in x for x in validate_manifest(path)))

    def test_corrupt_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text('{')
            self.assertTrue(validate_manifest(path))

    def test_invalid_record_type_rejected(self):
        self.assertTrue(validate_record(None, 0))


if __name__ == '__main__':
    unittest.main()
