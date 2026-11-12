"""Unit tests, not a neural inference benchmark."""
import numpy as np
from evaluate_tusimple_onnx import truth_mask,extract_binary_mask,score
record={'h_samples':[4,10,16],'lanes':[[5,6,7],[-2,-2,-2]]}
gt=truth_mask(record,(24,32),2)
assert gt.sum()>0 and score(gt,gt)[3:]==(1.,1.,1.)
assert score(np.zeros_like(gt),gt)[4]==0.
a=np.zeros((1,1,12,16),np.float32);a[0,0,2:6,2:5]=1
assert extract_binary_mask([a],(24,32)).sum()>0
try:extract_binary_mask([np.zeros((1,3,12,16))],(24,32))
except ValueError:pass
else:raise AssertionError('Invalid checkpoint output accepted')
# Output semantics must not be guessed from the min/max of one model output.
zero_logits=np.zeros((1,1,3,3),np.float32)
assert extract_binary_mask([zero_logits],(3,3),threshold=.5,output_semantics='logits').sum()==9
assert extract_binary_mask([zero_logits],(3,3),threshold=.5,output_semantics='probability').sum()==0
try:extract_binary_mask([-.1*np.ones((1,1,3,3),np.float32)],(3,3),output_semantics='probability')
except ValueError:pass
else:raise AssertionError('Out-of-range probability accepted')
try:extract_binary_mask([zero_logits],(3,3),output_semantics='auto')
except ValueError:pass
else:raise AssertionError('Ambiguous output semantics accepted')
print('PASS: TuSimple label rasterization, scores, output shapes and explicit checkpoint semantics')
