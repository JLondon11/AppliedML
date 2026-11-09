"""Numerical regression checks for the Chapter 9 DDPM implementation."""
import sys
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"chapters/generative_ai_part_i/figure_generation/pass12"))
from ddpm_digits_experiment import T, NoisePredictor, schedule, q_sample, sample_ddpm


class DDPMTests(unittest.TestCase):
    def setUp(self):
        self.beta,self.alpha,self.abar=schedule()

    def test_schedule(self):
        self.assertEqual(len(self.beta),T)
        self.assertTrue(bool(torch.all(self.beta>0)))
        self.assertTrue(bool(torch.all(self.alpha>0)))
        self.assertTrue(bool(torch.all(self.abar[1:]<self.abar[:-1])))

    def test_forward_noising_exact(self):
        x=torch.ones((3,64))
        eps=torch.zeros_like(x)
        t=torch.tensor([0,17,99])
        actual=q_sample(x,t,eps,self.abar)
        expected=torch.sqrt(self.abar[t])[:,None].expand_as(x)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7))

    def test_model_dimensions_and_gradients(self):
        torch.manual_seed(4)
        model=NoisePredictor()
        result=model(torch.randn(7,64),torch.arange(7))
        self.assertEqual(tuple(result.shape),(7,64))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_reverse_sampler_finite_and_seeded(self):
        torch.manual_seed(4)
        model=NoisePredictor(hidden=32)
        first=sample_ddpm(model,3,self.beta,self.alpha,self.abar,seed=19)
        second=sample_ddpm(model,3,self.beta,self.alpha,self.abar,seed=19)
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(torch.equal(first,second))


if __name__=='__main__':
    unittest.main()
