"""Scientific checks for actual digits scaling experiment."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/
 "chapters/generative_ai_part_ii/code/figures01_02_real_scaling"))
from experiment import run

class ScalingTests(unittest.TestCase):
    def test_reject_zero_epochs(self):
        with self.assertRaises(ValueError):run(Path("/tmp/unused-scaling"),epochs=0)
    def test_reject_single_seed(self):
        with self.assertRaises(ValueError):run(Path("/tmp/unused-scaling"),epochs=1,seeds=(11,))
if __name__=="__main__":
    unittest.main()
