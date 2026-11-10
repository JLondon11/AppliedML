"""Regression tests for real-image/label-word cross-modal alignment."""
import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"chapters/generative_ai_part_ii/code/figure08_digit_word_alignment"))
from experiment import ImageLabelAligner,TOKENS

class AlignmentTests(unittest.TestCase):
    def test_normalized_embeddings_and_scores(self):
        torch.manual_seed(8)
        model=ImageLabelAligner()
        logits,images,words=model(torch.rand(4,64))
        self.assertEqual(tuple(logits.shape),(4,10))
        self.assertEqual(tuple(words.shape),(10,32))
        self.assertTrue(torch.allclose(images.norm(dim=1),torch.ones(4),atol=1e-6))
        self.assertTrue(torch.allclose(words.norm(dim=1),torch.ones(10),atol=1e-6))
        self.assertEqual(TOKENS.shape[0],10)

    def test_trainable_both_encoders(self):
        model=ImageLabelAligner()
        logits,_,_=model(torch.rand(6,64))
        torch.nn.functional.cross_entropy(logits,torch.arange(6)).backward()
        self.assertIsNotNone(model.image[0].weight.grad)
        self.assertIsNotNone(model.characters.weight.grad)
        self.assertIsNotNone(model.words[0].weight.grad)

if __name__=="__main__":
    unittest.main()
