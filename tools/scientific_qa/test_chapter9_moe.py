"""Top-2 sparse expert routing numerical checks."""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"chapters/generative_ai_part_ii/code/figure05_sparse_experts"))
from experiment import MoE

class MoETests(unittest.TestCase):
    def test_top_two_and_normalization(self):
        model=MoE()
        x=torch.rand(11,64)
        logits,prob,sparse,selected=model(x,details=True)
        self.assertEqual(tuple(logits.shape),(11,10))
        self.assertTrue(bool(torch.all((sparse>0).sum(-1)==2)))
        self.assertTrue(torch.allclose(sparse.sum(-1),torch.ones(11),atol=1e-6))
        self.assertTrue(torch.allclose(prob.sum(-1),torch.ones(11),atol=1e-6))
    def test_gate_expert_gradients(self):
        model=MoE()
        result=model(torch.rand(16,64))
        torch.nn.functional.cross_entropy(result,torch.arange(16)%10).backward()
        self.assertTrue(model.gate.weight.grad is not None)
        self.assertTrue(any(p.grad is not None for e in model.experts for p in e.parameters()))
if __name__=="__main__":
    unittest.main()
