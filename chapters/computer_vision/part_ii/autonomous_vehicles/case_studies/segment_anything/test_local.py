"""Algorithm tests for SAM/COCO evaluator, not learned-model inference."""
import tempfile
from pathlib import Path
import numpy as np
from evaluate_sam_coco import xywh_to_xyxy,mask_scores,grabcut_box_baseline,sha256_file
def main():
    assert np.allclose(xywh_to_xyxy([2,3,10,4],30,40),[2,3,12,7])
    assert np.allclose(xywh_to_xyxy([-2,-1,10,10],30,40),[0,0,8,9])
    for box in ([0,0,0,2],[99,99,4,4]):
        try:xywh_to_xyxy(box,30,40)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    a=np.zeros((10,10),bool);a[2:6,2:6]=True
    b=np.zeros((10,10),bool);b[4:8,4:8]=True
    assert mask_scores(a,a)==(1.,1.)
    assert np.allclose(mask_scores(a,b),[4/28,8/32])
    assert mask_scores(np.zeros((2,2)),np.zeros((2,2)))==(1.,1.)
    try:mask_scores(np.zeros((2,2)),np.zeros((3,3)))
    except ValueError:pass
    else:raise AssertionError('Mismatched mask accepted')
    img=np.zeros((90,120,3),dtype=np.uint8)
    img[:]=[30,40,60];img[25:60,40:80]=[90,180,230]
    assert grabcut_box_baseline(img,np.array([28,15,94,73]),2).shape==(90,120)
    with tempfile.TemporaryDirectory() as td:
        f=Path(td)/'test';f.write_text('SAM')
        assert len(sha256_file(f))==64
    print('PASS: COCO boxes, mask IoU/Dice, GrabCut and provenance hash')
if __name__=='__main__':main()
