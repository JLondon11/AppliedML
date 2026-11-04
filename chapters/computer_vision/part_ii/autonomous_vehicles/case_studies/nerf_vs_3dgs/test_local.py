"""Data-contract checks only; no claim to trained model inference."""
import tempfile,json
from pathlib import Path
from run_nerfstudio_pair import parse_metrics,verify_scene
def main():
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)
        records=[{'file_path':f'images/{i:04d}.png','transform_matrix':
          [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]} for i in range(12)]
        (path/'transforms.json').write_text(json.dumps({'frames':records,'train_filenames':[r['file_path'] for r in records[:10]],'test_filenames':[r['file_path'] for r in records[10:]]}))
        try:verify_scene(path)
        except FileNotFoundError:pass
        else:raise AssertionError('Real original frames should be required')
        (path/'images').mkdir()
        for i in range(12):(path/'images'/f'{i:04d}.png').write_bytes(b'fixture')
        assert verify_scene(path)[0]==12
        (path/'eval.json').write_text(json.dumps({'results':{'psnr':25.5,'ssim':.82,'lpips':.12}}))
        assert parse_metrics(path/'eval.json')['psnr']==25.5
    print('PASS real-scene input contract, missing frame rejection, ns-eval output shape')
if __name__=='__main__':main()
