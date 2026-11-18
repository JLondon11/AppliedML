"""Software-only regression: never substitute archived digit counts into a new run.

Fixture modifications are NOT empirical generated model outputs. Tests the
plot/verification software against an archived record and an altered synthetic
fixture. No generated-fixture statistics belong in textbook experiments.
"""
import contextlib,csv,io,tempfile,unittest
from pathlib import Path
from plot_verified_original_class_counts import main
HERE=Path(__file__).resolve().parent

class DigitFigureCurrentRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (HERE/'figure_6_44_class_counts_pass27.csv').open(newline='') as f:
            cls.archived=list(csv.DictReader(f))
        if len(cls.archived)!=22:raise AssertionError('Missing actual 22 archived digit count records')
    def save(self,rows,folder):
        src=Path(folder)/'counts.csv'
        with src.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['arm','label','correct','n'])
            writer.writeheader();writer.writerows(rows)
        return src
    def test_original_pass27_strict_replay(self):
        with tempfile.TemporaryDirectory() as d:
            src=self.save(self.archived,d);output=Path(d)/'fig'
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                main(src,output,expected_conditional=778,expected_unconditional=94)
            self.assertTrue((output/'original_digit_classwise_scientific.png').is_file())
            self.assertIn('778/1000',stream.getvalue())
    def test_changed_count_uses_current_fixture_not_archived_count(self):
        # Software fixture only: represent a rerun that differs by one scored digit.
        rows=[dict(r) for r in self.archived]
        overall=next(r for r in rows if r['arm']=='conditional' and r['label']=='overall')
        part=next(r for r in rows if r['arm']=='conditional' and r['label']!='overall' and int(r['correct'])<100)
        overall['correct']=str(int(overall['correct'])+1)
        part['correct']=str(int(part['correct'])+1)
        with tempfile.TemporaryDirectory() as d:
            src=self.save(rows,d);output=Path(d)/'fig';buffer=io.StringIO()
            with contextlib.redirect_stdout(buffer):main(src,output)
            self.assertIn('779/1000',buffer.getvalue())
            self.assertTrue((output/'original_digit_classwise_scientific.png').stat().st_size>10000)
            with self.assertRaisesRegex(ValueError,'Archived conditional count'):
                main(src,Path(d)/'strict',expected_conditional=778)
    def test_inconsistent_overall_rejected(self):
        rows=[dict(r) for r in self.archived]
        overall=next(r for r in rows if r['arm']=='conditional' and r['label']=='overall')
        overall['correct']=str(int(overall['correct'])+1)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'overall'):
                main(self.save(rows,d),Path(d)/'bad')

if __name__=='__main__':unittest.main()
