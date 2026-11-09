"""Numerical regression tests for classifier-feature quality diagnostics."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/
  "chapters/generative_ai_part_i/figure_generation/pass12"))
from quality_benchmark import domain_frechet, class_entropy


class QualityBenchmarkTests(unittest.TestCase):
    def test_identity_distance(self):
        x=np.array([[1.,2.],[3.,4.],[5.,8.],[2.,7.]])
        self.assertAlmostEqual(domain_frechet(x,x),0,places=5)

    def test_shifted_distributions(self):
        x=np.array([[1.,2.],[3.,4.],[5.,8.],[2.,7.]])
        self.assertGreater(domain_frechet(x,x+3),1)

    def test_singular_covariance(self):
        self.assertAlmostEqual(domain_frechet(np.ones((10,4)),
                                                np.zeros((10,4))),4.,places=4)

    def test_entropy_extremes(self):
        self.assertAlmostEqual(class_entropy(np.zeros(100,dtype=int)),0,places=6)
        self.assertAlmostEqual(class_entropy(np.repeat(np.arange(10),10)),
                               np.log(10),places=6)


if __name__=="__main__":
    unittest.main()
