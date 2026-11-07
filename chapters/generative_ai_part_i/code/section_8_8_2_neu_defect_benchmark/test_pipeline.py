"""Fast end-to-end TEST-ONLY smoke fixture; these are not NEU experimental results."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
CLASSES=('crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches')

def main():
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp);dataset=path/'fixture';out=path/'output'
        for c,name in enumerate(CLASSES):
            (dataset/name).mkdir(parents=True)
            for i in range(10):
                # Deliberately artificial test-only data that must never be reported as a NEU benchmark.
                arr=np.random.default_rng(c*1000+i).integers(0,256,(64,64),dtype=np.uint8)
                Image.fromarray(arr).save(dataset/name/f'{name}_{i:03}.png')
        cmd=[sys.executable,str(ROOT/'run.py'),'--data-root',str(dataset),'--output',str(out),
             '--smoke-test','--gan-steps','2','--classifier-epochs','1',
             '--generated-per-class','2','--batch-size','8']
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['smoke_fixture_mode'] is True
        assert manifest['real_NEU_results_claimed'] is False
        assert manifest['split_sizes']=={'train':36,'validation':12,'test':12}
        with (out/'dataset_manifest.csv').open() as file:rows=list(csv.DictReader(file))
        assert len(rows)==60 and len({r['decoded_sha256'] for r in rows})==60
        with (out/'benchmark_metrics.csv').open() as file:metrics=list(csv.DictReader(file))
        assert set(r['arm'] for r in metrics)=={'real_only','gan_augmented'}
        with (out/'test_predictions.csv').open() as file:pred=list(csv.DictReader(file))
        assert len(pred)==24
        for img in ('conditional_gan_samples.png','test_confusion_matrices.png','validation_balanced_accuracy.png'):
            assert (out/'figures'/img).stat().st_size > 8000
        for weight in ('generator.pt','classifier_real_only.pt','classifier_gan_augmented.pt'):
            assert (out/weight).stat().st_size > 10000
        # A second command without --smoke-test must reject the synthetic 10-per-class fixture.
        disallowed=subprocess.run(cmd[:cmd.index('--smoke-test')]+cmd[cmd.index('--smoke-test')+1:],
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        assert disallowed.returncode != 0
    print('PASS: synthetic fixture smoke test, anti-mislabel gate, generated figures, outputs, and checkpoints')

if __name__=='__main__': main()
