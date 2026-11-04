"""Focused tests of geometry, required prediction fields and KITTI frame IDs."""
import numpy as np
from render_verified_pointpillars import corners_bev,parse_pred
def run():
    pts=corners_bev([5.,4.,0.,4.,2.,1.,0.])
    assert pts.shape==(5,2)
    assert np.allclose(pts[0],[3,3])
    assert np.allclose(pts[0],pts[-1])
    z=corners_bev([0,0,0,2,4,1,np.pi/2])
    assert np.allclose(z[0],[2,-1],atol=1e-8)
    frame,boxes,scores,names=parse_pred({'frame_id':'8','boxes_lidar':np.array([[0,0,0,2,2,2,0]]),'name':np.array(['Car']),'score':np.array([.9])})
    assert frame=='000008' and boxes.shape==(1,7) and len(scores)==1
    try:parse_pred({'frame_id':'8','score':np.array([]),'name':np.array([])})
    except ValueError:pass
    else:raise AssertionError('Must fail closed if missing model geometry')
    print('PASS: oriented box geometry, frame IDs, output checks, no fabricated predictions')
if __name__=='__main__':run()
