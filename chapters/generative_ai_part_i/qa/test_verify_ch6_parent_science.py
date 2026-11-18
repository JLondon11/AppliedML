"""Unit tests use deliberate software fixtures, never claimed as experiments."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from verify_ch6_parent_science import audit


class ParentCertificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'chapters/generative_ai_part_i'
        (self.base / 'quality_assurance').mkdir(parents=True)
        (self.base / 'evidence').mkdir()
        for name in ('data.csv', 'train.py', 'predictions.csv', 'metrics.json', 'review.md'):
            (self.base / 'evidence' / name).write_text('test fixture only')
        fields = {
            'dataset_provenance': 'data.csv', 'executed_model_source': 'train.py',
            'original_model_outputs': 'predictions.csv', 'quantitative_evaluation': 'metrics.json',
            'scientific_review_record': 'review.md'
        }
        self.parents = []
        self.figures = []
        for i in range(1, 15):
            self.parents.append({
                'unit_id': f'p{i:02d}', 'exact_manuscript_title': f'Test parent {i}',
                'scope_reviewed': True, 'title_claim_match_verified': True,
                'scientific_status': 'SCIENTIFIC_CERTIFIED', 'level': 'B',
                'r1_r10': {f'R{n}': 'PASS' for n in range(1, 11)},
                'code_commit': 'a'*40, 'figure_ids': [f'6.{i}'],
                **{k: f'chapters/generative_ai_part_i/evidence/{v}' for k, v in fields.items()}
            })
            self.figures.append({'figure_id': f'6.{i}', 'kind': 'experimental',
                                 'scientific_reviewed': True, 'dataset_provenance_verified': True})
        self.inv = {'compiled_source_reviewed': True,
                    'parent_applications': self.parents, 'figures': self.figures}
        self.path = self.base / 'quality_assurance/production_inventory.json'

    def run_audit(self, data=None):
        self.path.write_text(json.dumps(self.inv if data is None else data))
        return audit(self.root)

    def test_valid_fixture_is_not_scientific_certification(self):
        report = self.run_audit()
        self.assertEqual(report['certified_parent_applications'],14)
        self.assertEqual(report['status'],'PARENT_SCOPE_RECORDS_COMPLETE_INDEPENDENT_REVIEW_REQUIRED')

    def test_missing_inventory_blocks(self):
        self.assertEqual(audit(self.root)['status'], 'BLOCKED')

    def test_parent_on_scientific_hold_blocks(self):
        doc=copy.deepcopy(self.inv);doc['parent_applications'][0]['scientific_status']='SCIENTIFIC_HOLD'
        self.assertEqual(self.run_audit(doc)['certified_parent_applications'],13)

    def test_12_scoped_certificates_cannot_replace_14_parents(self):
        doc=copy.deepcopy(self.inv);doc['parent_applications']=doc['parent_applications'][:12]
        self.assertEqual(self.run_audit(doc)['status'],'BLOCKED')

    def test_level_a_embedded_demo_not_parent_approval(self):
        doc=copy.deepcopy(self.inv);doc['parent_applications'][0]['level']='A'
        self.assertEqual(self.run_audit(doc)['status'],'BLOCKED')

    def test_claim_title_mismatch_blocks(self):
        doc=copy.deepcopy(self.inv);doc['parent_applications'][0]['title_claim_match_verified']=False
        self.assertEqual(self.run_audit(doc)['status'],'BLOCKED')

    def test_absent_original_outputs_blocks(self):
        doc=copy.deepcopy(self.inv);doc['parent_applications'][0]['original_model_outputs']='missing.csv'
        self.assertEqual(self.run_audit(doc)['status'],'BLOCKED')

    def test_conceptual_figure_cannot_replace_experimental(self):
        doc=copy.deepcopy(self.inv);doc['figures'][0]['kind']='conceptual'
        self.assertEqual(self.run_audit(doc)['status'],'BLOCKED')

    def test_inapplicable_fundamental_requirement_blocks(self):
        doc=copy.deepcopy(self.inv);doc['parent_applications'][0]['r1_r10']['R8']='N/A — JUSTIFIED'
        self.assertEqual(self.run_audit(doc)['status'],'BLOCKED')

    def test_legitimate_nonfundamental_na_needs_justification(self):
        doc=copy.deepcopy(self.inv);doc['parent_applications'][0]['r1_r10']['R2']='N/A — JUSTIFIED'
        self.assertEqual(self.run_audit(doc)['status'],'BLOCKED')
        doc['parent_applications'][0]['na_justifications']={'R2':'This evaluation uses a fixed, preexisting numerical solver and independent check.'}
        self.assertEqual(self.run_audit(doc)['status'],'PARENT_SCOPE_RECORDS_COMPLETE_INDEPENDENT_REVIEW_REQUIRED')

    def test_missing_source_commit_blocks(self):
        doc=copy.deepcopy(self.inv);doc['parent_applications'][0]['code_commit']='not-a-hash'
        self.assertEqual(self.run_audit(doc)['status'],'BLOCKED')

    def test_duplicate_parent_ids_block(self):
        doc=copy.deepcopy(self.inv);doc['parent_applications'][0]['unit_id']='p02'
        self.assertEqual(self.run_audit(doc)['status'],'BLOCKED')


if __name__ == '__main__':unittest.main()
