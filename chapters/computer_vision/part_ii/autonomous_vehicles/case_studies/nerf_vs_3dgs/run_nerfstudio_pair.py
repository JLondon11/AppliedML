"""Matched Nerfstudio Nerfacto/GS Splatfacto real-scene training + evaluation.

Requires Nerfstudio, CUDA, real posed image dataset. No fabricated benchmark data.
Nerfacto is not identical to original Mip-NeRF360 model.
"""
import argparse,csv,hashlib,json,shutil,subprocess,time
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def verify_scene(scene):
    t=scene/'transforms.json'
    if not t.is_file():raise FileNotFoundError('Original posed-scene transforms.json required')
    data=json.loads(t.read_text());frames=data.get('frames',[])
    if len(frames)<12:raise ValueError('A real, >=12 frame scene is required')
    train=set(data.get('train_filenames',[]))
    test=set(data.get('test_filenames',[]))
    if not train or not test or train & test:
        raise ValueError('Require explicit disjoint train_filenames and test_filenames')
    known={item.get('file_path') for item in frames}
    if not (train | test).issubset(known):
        raise ValueError('Train/test split includes file not in dataset')
    for item in frames:
        if not item.get('file_path') or 'transform_matrix' not in item:
            raise ValueError('Missing image or pose')
        if not (scene/item['file_path']).is_file():
            raise FileNotFoundError('Original image unavailable: '+item['file_path'])
    return len(frames),sha256(t)

def parse_metrics(file):
    d=json.loads(file.read_text());m=d.get('results',d)
    scores={k:float(m[k]) for k in ('psnr','ssim','lpips') if isinstance(m.get(k),(float,int))}
    if 'psnr' not in scores:raise ValueError('No genuine ns-eval PSNR returned')
    return scores

def run(a):
    n,transform_sha=verify_scene(a.scene)
    for tool in ('ns-train','ns-eval'):
        if shutil.which(tool) is None:raise FileNotFoundError('Nerfstudio CLI is missing: '+tool)
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for model in ('nerfacto','splatfacto'):
        out=a.output/model;out.mkdir(parents=True,exist_ok=True)
        train=['ns-train',model,'--output-dir',str(out/'trained'),'nerfstudio-data',
               '--data',str(a.scene),'--eval-mode','filename']
        start=time.perf_counter()
        with (out/'train.log').open('w') as log:
            subprocess.run(train,stdout=log,stderr=subprocess.STDOUT,check=True)
        duration=time.perf_counter()-start
        configs=sorted((out/'trained').rglob('config.yml'))
        if len(configs)!=1:raise RuntimeError(f'{model}: expected one config, got {len(configs)}')
        output=out/'eval.json'
        evaluate=['ns-eval','--load-config',str(configs[0]),
                  '--output-path',str(output),'--render-output-path',str(out/'rendered')]
        with (out/'eval.log').open('w') as log:
            subprocess.run(evaluate,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not output.is_file():raise RuntimeError('Real ns-eval file absent')
        metrics=parse_metrics(output)
        rows.append([model,duration,metrics['psnr'],metrics.get('ssim'),metrics.get('lpips')])
        (out/'commands.json').write_text(json.dumps({'train':train,'eval':evaluate},indent=2))
    with (a.output/'model_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','training_seconds','test_PSNR_dB','test_SSIM','test_LPIPS']);w.writerows(rows)
    (a.output/'provenance.json').write_text(json.dumps({
       'frames':n,'transforms_sha256':transform_sha,
       'protocol':'same user-supplied real scene and filename-defined test split for both models',
       'limits':'No original Mip-NeRF360 method, no model-specific published benchmark rerun, no FPS metric'},indent=2))
    print(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a)
if __name__=='__main__':main()
