"""Regression tests for CSV schema corruption; synthetic rows are QA fixtures only."""
import csv,tempfile,unittest
from pathlib import Path
from validate_scoped_csv_schema import FIELDS,validate

class ScopedRegisterSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'scoped.csv'
    def rows(self):
        observations=[]
        for i in range(1,13):
            observations.append([f'CH6-D{i:02d}','fixture title',
                'B' if i==4 else 'A','fixture scope',*(['PASS']*10),
                'SCIENTIFIC_CERTIFIED','PRODUCTION_HOLD','fixture limitation'])
        return observations
    def save(self,rows):
        with self.path.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(FIELDS);w.writerows(rows)
    def test_valid_fixture_is_only_structural(self):
        self.save(self.rows())
        report=validate(self.path)
        self.assertEqual(report['units'],12)
        self.assertEqual(report['production_release'],'HOLD')
        self.assertEqual(report['parent_application_certified_by_this_register'],0)
    def test_extra_pass_shifts_scientific_status_and_is_rejected(self):
        rows=self.rows();rows[-1].insert(14,'PASS')
        self.save(rows)
        with self.assertRaisesRegex(ValueError,'17 fields'):
            validate(self.path)
    def test_missing_pass_rejected(self):
        rows=self.rows();rows[3][4]='PARTIAL'
        self.save(rows)
        with self.assertRaisesRegex(ValueError,'requirement'):
            validate(self.path)
    def test_missing_row_rejected(self):
        self.save(self.rows()[:-1])
        with self.assertRaisesRegex(ValueError,'12 or more'):
            validate(self.path)
    def test_gapped_identifier_rejected(self):
        rows=self.rows();rows[4][0]='CH6-D06'
        self.save(rows)
        with self.assertRaisesRegex(ValueError,'continuity'):
            validate(self.path)
    def test_falsely_approved_print_rejected(self):
        rows=self.rows();rows[-1][-2]='PRODUCTION_APPROVED'
        self.save(rows)
        with self.assertRaisesRegex(ValueError,'production approval'):
            validate(self.path)

if __name__=='__main__':unittest.main()
