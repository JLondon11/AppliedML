"""Chapter 9 real-input KV-cache correctness tests."""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/
    "chapters/generative_ai_part_ii/code/figure04_kv_cache"))
from experiment import CachedRowPredictor

class TestCache(unittest.TestCase):
    def test_cached_is_equal_to_causal_full(self):
        torch.manual_seed(2)
        model=CachedRowPredictor()
        model.eval()
        x=torch.rand(7,7,8)
        with torch.no_grad():
            delta=(model.full(x)-model.cached(x)).abs().max().item()
        self.assertLess(delta,1e-5)

    def test_training_gradient(self):
        model=CachedRowPredictor()
        x=torch.rand(4,7,8)
        torch.nn.functional.mse_loss(model.full(x),torch.zeros_like(x)).backward()
        for layer in (model.q,model.k,model.v):
            self.assertTrue(all(p.grad is not None for p in layer.parameters()))

if __name__=="__main__":
    unittest.main()
