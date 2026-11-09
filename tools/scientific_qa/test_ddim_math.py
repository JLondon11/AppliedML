"""Regression tests for deterministic DDIM on the trained-real-digits schedule."""
import sys
import unittest
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parents[2] / "chapters/generative_ai_part_i/figure_generation/pass12"
sys.path.insert(0, str(HERE))
from ddim_ddpm_comparison import ddim_timesteps, sample_ddim
from ddpm_digits_experiment import NoisePredictor, schedule


class TestDDIM(unittest.TestCase):
    def test_timestep_grid(self):
        for count in (20, 50, 100):
            ts = ddim_timesteps(count)
            self.assertEqual(len(ts), count)
            self.assertEqual(ts[0], 99)
            self.assertEqual(ts[-1], 0)
            self.assertTrue(all(x > y for x, y in zip(ts, ts[1:])))

    def test_invalid_timestep_count(self):
        for count in (0, 1, 101):
            with self.assertRaises(ValueError):
                ddim_timesteps(count)

    def test_deterministic_sampler_is_finite(self):
        torch.manual_seed(2)
        model = NoisePredictor(hidden=32)
        z = torch.randn(5, 64)
        alpha_bar = schedule()[2]
        a = sample_ddim(model, z, alpha_bar, 20)
        b = sample_ddim(model, z, alpha_bar, 20)
        self.assertTrue(bool(torch.isfinite(a).all()))
        self.assertTrue(torch.equal(a, b))

    def test_analytical_zero_predictor(self):
        class Zero(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)
        alpha_bar = schedule()[2]
        z = torch.randn(3, 64)
        result = sample_ddim(Zero(), z, alpha_bar, 20)
        self.assertTrue(torch.allclose(result, z/torch.sqrt(alpha_bar[-1]), atol=2e-4))


if __name__ == "__main__":
    unittest.main()
