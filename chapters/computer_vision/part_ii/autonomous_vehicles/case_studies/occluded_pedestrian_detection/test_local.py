"""Geometry-only unit tests; not trained model inference."""
import numpy as np
from evaluate_crowdhuman_frcnn import annotations,overlap,assign_recall
r={'gtboxes':[{'tag':'person','fbox':[1,2,10,20],'vbox':[1,2,8,10],
 'extra':{'ignore':0}},{'tag':'mask','fbox':[1,2,10,20],
 'vbox':[1,2,10,20],'extra':{'ignore':1}}]}
gt=annotations(r)
assert len(gt)==1 and abs(gt[0][1]-.6)<1e-10
assert overlap([[0,0,10,10]],[[0,0,10,10]])[0,0]==1
assert overlap([[0,0,1,1]],[[2,2,3,3]])[0,0]==0
assert assign_recall([(gt[0][0],.9)],gt).tolist()==[True]
assert assign_recall([],gt).tolist()==[False]
print('PASS: CrowdHuman annotation, occlusion proxy, local match')
