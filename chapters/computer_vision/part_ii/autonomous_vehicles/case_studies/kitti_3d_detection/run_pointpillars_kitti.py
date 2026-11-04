"""Run the real OpenPCDet PointPillars checkpoint on official KITTI validation.

This script deliberately fails closed if the real dataset, checkpoint or GPU are missing.
There are NO synthetic predictions or fabricated benchmark results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(pcdet,kitti,checkpoint,config):
    missing=[]
    for f in [pcdet/'tools'/'test.py', config, checkpoint,
              kitti/'ImageSets'/'val.txt',kitti/'kitti_infos_val.pkl',
              kitti/'training'/'velodyne',kitti/'training'/'calib',
              kitti/'training'/'label_2']:
        if not f.exists():missing.append(str(f))
    if missing:raise FileNotFoundError('Missing required official inputs:\n'+'\n'.join(missing))
    ids=[x.strip() for x in (kitti/'ImageSets'/'val.txt').read_text().splitlines() if x.strip()]
    if not ids:raise ValueError('Empty KITTI validation split')
    for id in ids[:min(5,len(ids))]:
        if not (kitti/'training'/'velodyne'/f'{id}.bin').is_file():
            raise FileNotFoundError(f'KITTI validation LiDAR frame missing: {id}')
    if checkpoint.stat().st_size<1024*1024:
        raise ValueError('Checkpoint suspiciously small; verify the actual model weight file')
    return {'val_frames':len(ids),'checkpoint_sha256':sha256(checkpoint),
            'config_sha256':sha256(config)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--openpcdet',type=Path,required=True)
    ap.add_argument('--kitti',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('verified_run'))
    ap.add_argument('--dry-run',action='store_true',help='validate local dataset/weights only')
    args=ap.parse_args()
    root=args.openpcdet.expanduser().resolve()
    data=args.kitti.expanduser().resolve()
    ckpt=args.checkpoint.expanduser().resolve()
    config=root/'tools'/'cfgs'/'kitti_models'/'pointpillar.yaml'
    details=validate_inputs(root,data,ckpt,config)
    link=root/'data'/'kitti'
    if not link.exists() or link.resolve()!=data:
        raise RuntimeError(f'OpenPCDet expects KITTI at {link}, but it is not linked to {data}; create a symlink manually, then retry')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'dataset':'KITTI 3D object detection official training split with local val.txt',
              'framework':'OpenPCDet', 'model':'PointPillars',
              'dataset_root':str(data),'checkpoint':str(ckpt),
              'framework_repo':str(root),'config':str(config),**details}
    manifest_path=args.output/'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if args.dry_run:
        print('Dataset and checkpoint input verification passed; no model inference executed.')
        print(manifest_path);return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('OpenPCDet PointPillars requires a configured CUDA GPU; no run performed')
    cmd=[sys.executable,'test.py','--cfg_file','cfgs/kitti_models/pointpillar.yaml',
         '--batch_size','1','--workers','2','--ckpt',str(ckpt),
         '--eval_tag','chapter5_pointpillars_verified','--save_to_file']
    manifest['command']=cmd
    manifest['start_unix_time']=time.time()
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    env=os.environ.copy()
    with (args.output/'run_stdout_stderr.log').open('w') as log:
        subprocess.run(cmd,cwd=root/'tools',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest['run_status']='completed'
    manifest['end_unix_time']=time.time()
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Evaluation finished. Inspect OpenPCDet output/eval result.pkl and the full log.')


if __name__=='__main__':main()
