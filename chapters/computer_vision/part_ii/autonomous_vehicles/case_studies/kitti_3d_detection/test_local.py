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

    bad_cases=[
        {'frame_id':'8','boxes_lidar':np.array([[0,0,0,-2,2,2,0]]),'name':['Car'],'score':[.8]},
        {'frame_id':'8','boxes_lidar':np.array([[0,0,0,2,2,2,0]]),'name':['Car'],'score':[float('nan')]},
        {'frame_id':'8','boxes_lidar':np.array([[0,0,0,2,2,2,0]]),'name':['Car'],'score':[1.1]},
    ]
    for invalid in bad_cases:
        try:parse_pred(invalid)
        except ValueError:pass
        else:raise AssertionError('Accepted scientifically invalid 3D model prediction')
    # Preflight-only fixture checks: no trained-model inference or scientific results.
    import tempfile
    from pathlib import Path
    from run_pointpillars_kitti import validate_inputs
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); pc=root/'OpenPCDet'; kit=root/'kitti'
        cfg=pc/'tools'/'cfgs'/'kitti_models'/'pointpillar.yaml'
        for item in (pc/'tools'/'test.py',cfg,kit/'kitti_infos_val.pkl'):
            item.parent.mkdir(parents=True,exist_ok=True)
            item.write_text('preflight test fixture')
        ckpt=root/'placeholder_not_a_model.pth'
        ckpt.write_bytes(b'0'*(1024*1024+1))
        split=kit/'ImageSets'/'val.txt';split.parent.mkdir(parents=True,exist_ok=True)
        split.write_text('000001\n000002\n000003\n000004\n000005\n000006\n')
        for frame in range(1,7):
            id=f'{frame:06d}'
            for folder,suffix,body in [('velodyne','.bin',b'0'*16),
                                       ('calib','.txt',b'P2: fixture'),
                                       ('label_2','.txt',b'Car fixture')]:
                path=kit/'training'/folder/(id+suffix)
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(body)
        result=validate_inputs(pc,kit,ckpt,cfg)
        assert result['val_frames']==6 and result['validated_original_label_files']==6
        assert len(result['val_split_sha256'])==64
        # A real zero-object KITTI scene may have an empty label file.
        empty_gt=kit/'training'/'label_2'/'000006.txt'
        empty_gt.write_text('')
        assert validate_inputs(pc,kit,ckpt,cfg)['val_frames']==6
        # A truncated XYZI file must be rejected before GPU execution.
        damaged=kit/'training'/'velodyne'/'000006.bin'
        damaged.write_bytes(b'0'*17)
        try:validate_inputs(pc,kit,ckpt,cfg)
        except ValueError as e:assert 'XYZI' in str(e)
        else:raise AssertionError('Accepted truncated original LiDAR')
        damaged.write_bytes(b'0'*16)
        missing=kit/'training'/'velodyne'/'000006.bin'
        missing.unlink()
        try:validate_inputs(pc,kit,ckpt,cfg)
        except FileNotFoundError as e:assert '000006' in str(e)
        else:raise AssertionError('Failed to check late split item')
        missing.write_bytes(b'0'*16)
        split.write_text('000001\n000001\n')
        try:validate_inputs(pc,kit,ckpt,cfg)
        except ValueError as e:assert 'Duplicate' in str(e)
        else:raise AssertionError('Failed to reject duplicated split')
        split.write_text('../001\n')
        try:validate_inputs(pc,kit,ckpt,cfg)
        except ValueError as e:assert 'six-digit' in str(e)
        else:raise AssertionError('Failed to reject malformed split ID')
    print('PASS: oriented box geometry, frame IDs, output checks, no fabricated predictions')
if __name__=='__main__':run()
