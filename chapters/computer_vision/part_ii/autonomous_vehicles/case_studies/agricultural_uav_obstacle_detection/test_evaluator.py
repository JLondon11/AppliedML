"""Algorithm-only unit checks; do not count as VisDrone real-model results."""
import tempfile
from pathlib import Path
import numpy as np
from evaluate_visdrone_yolo import iou,parse_visdrone_annotation,ap50_for_class
def main():
    assert np.isclose(iou(np.array([0.,0.,10.,10.]),[[0,0,10,10]])[0],1)
    assert iou(np.array([0.,0.,1.,1.]),[[2,2,4,4]])[0]==0
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'one.txt'
        p.write_text('10,20,30,40,1,1,0,0\n0,0,8,8,0,0,0,0\n')
        gt,ignored=parse_visdrone_annotation(p)
        assert gt.shape==(1,5) and ignored.shape==(1,4) and gt[0,4]==0
    records=[{'id':'one.jpg','gt':np.array([[0,0,10,10,0.]]),
      'pred':np.array([[0,0,10,10,.9,0.]]),'ignored':np.empty((0,4))}]
    assert ap50_for_class(records,0)[0]==1.
    records[0]['pred']=np.array([[20,20,30,30,.9,0.]])
    assert ap50_for_class(records,0)[0]==0.
    print('PASS: IoU, parsing, ignored labels, AP50 positive/negative fixture')
if __name__=='__main__':main()
