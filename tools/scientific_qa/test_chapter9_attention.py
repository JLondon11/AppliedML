"""Local validation for Figure 3 model math and actual held-out outputs."""
import sys
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"chapters/generative_ai_part_ii/code/figure03_scaled_attention"))
from experiment import RowAttentionClassifier

class TestAttention(unittest.TestCase):
    def test_row_attention_weights_normalized(self):
        torch.manual_seed(1)
        model=RowAttentionClassifier()
        logits,w,s,q,k,v=model(torch.rand(4,8,8),explain=True)
        self.assertEqual(tuple(logits.shape),(4,10))
        self.assertEqual(tuple(w.shape),(4,8,8))
        self.assertTrue(torch.allclose(w.sum(-1),torch.ones(4,8),atol=1e-6))
        self.assertTrue(torch.allclose(s,q@k.transpose(-2,-1)/32**.5,atol=1e-6))

    def test_training_gradients(self):
        model=RowAttentionClassifier()
        logits=model(torch.rand(5,8,8))
        torch.nn.functional.cross_entropy(logits,torch.arange(5)).backward()
        for name in ('q','k','v'):
            self.assertTrue(any(p.grad is not None for p in getattr(model,name).parameters()))

if __name__=='__main__':
    unittest.main()
