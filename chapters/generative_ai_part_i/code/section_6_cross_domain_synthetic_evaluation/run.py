"""NEU vs photographic texture domain-shift experiment with actual saved GAN images.
Run: python run.py --neu-zip "/path/archive (2).zip" --gan-results /path/Pass13/results --output results
"""
import argparse,csv,hashlib,json,zipfile,io
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage import data,filters
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS=(20261106,20261107,20261108)
def features(images):
    vectors=[]
    for im in images:
        x=np.asarray(im,dtype=np.float32)
        if x.min()<0:x=(x+1)/2
        if x.max()>1:x/=255
        gx=np.diff(x,axis=1);gy=np.diff(x,axis=0)
        hist=np.histogram(x,bins=8,range=(0,1),density=True)[0]/8
        vectors.append([x.mean(),x.std(),np.abs(gx).mean(),np.abs(gy).mean(),
            filters.laplace(x).std(),np.mean(x*x),gx.std(),gy.std(),
            *np.quantile(x,[.1,.25,.5,.75,.9]),*hist])
    return np.asarray(vectors,dtype=float)

def read_manifest(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--neu-zip',type=Path,required=True)
    ap.add_argument('--gan-results',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    expected={r['filename']:r for r in read_manifest(a.gan_results/'dataset_manifest.csv')}
    source_train=[];source_test=[]
    with zipfile.ZipFile(a.neu_zip) as z:
        for path in z.namelist():
            if not path.lower().endswith('.jpg') or '/images/' not in path:continue
            rel=path.split('NEU-DET/',1)[-1]
            if rel not in expected:continue
            image=Image.open(io.BytesIO(z.read(path))).convert('L')
            pixels=np.asarray(image.resize((64,64),Image.Resampling.BILINEAR))
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==expected[rel]['decoded_sha256']
            split=expected[rel]['split']
            if split=='train':source_train.append(pixels.astype(np.float32)/255)
            elif split=='test':source_test.append(pixels.astype(np.float32)/255)
    assert len(source_train)==1080 and len(source_test)==360
    target_train=[];target_test=[]
    for key in ('brick','grass','gravel'):
        im=getattr(data,key)().astype(np.float32)/255
        for r in range(8):
            for c in range(8):
                tile=im[r*64:(r+1)*64,c*64:(c+1)*64]
                (target_train if r<5 else target_test).append(tile)
    assert (len(target_train),len(target_test))==(120,72)
    sx=features(source_train);testsource=features(source_test)
    tx=features(target_train);testtarget=features(target_test)
    train_source=sx[np.random.default_rng(944).choice(len(sx),120,replace=False)]
    Xtest=np.concatenate((testsource,testtarget))
    ytest=np.r_[np.zeros(360,dtype=int),np.ones(72,dtype=int)]
    rows=[];predictions=[];probabilities={}
    for seed in SEEDS:
        generated=torch.load(a.gan_results/f'generated_seed{seed}.pt',map_location='cpu',weights_only=True)
        gf=features(generated['x'].numpy()[:,0])
        sampled=gf[np.random.default_rng(seed).choice(len(gf),120,replace=False)]
        for arm,src in (('real_source',train_source),('gan_source',sampled)):
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200,C=1))
            model.fit(np.r_[src,tx],np.r_[np.zeros(120,dtype=int),np.ones(120,dtype=int)])
            prob=model.predict_proba(Xtest)[:,1]
            probabilities[(seed,arm)]=prob
            rows.append(dict(seed=seed,arm=arm,
                test_balanced_accuracy=float(balanced_accuracy_score(ytest,prob>=.5)),
                test_roc_auc=float(roc_auc_score(ytest,prob)),
                real_neu_source_recall=float(np.mean(prob[:360]<.5)),
                real_texture_target_recall=float(np.mean(prob[360:]>=.5)),
                n_source_train=120,n_target_train=120,n_source_test=360,n_target_test=72))
            predictions.extend((seed,arm,i,int(y),float(p)) for i,(y,p) in enumerate(zip(ytest,prob)))
    with (a.output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (a.output/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','arm','test_index','ground_truth_target_domain','probability_target_domain']);w.writerows(predictions)
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),constrained_layout=True)
    for j,key in enumerate(('test_balanced_accuracy','test_roc_auc')):
        for i,arm in enumerate(('real_source','gan_source')):
            values=np.array([r[key] for r in rows if r['arm']==arm])
            axes[j].scatter(np.array([-.08,0,.08])+i,values,s=45)
            axes[j].hlines(values.mean(),i-.16,i+.16,lw=2)
        axes[j].set(xticks=[0,1],xticklabels=['Real source','GAN source'],
                    ylabel=key.replace('_',' '),ylim=(0,1.03))
        axes[j].grid(axis='y',alpha=.15)
    fig.savefig(a.output/'domain_transfer_accuracy.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for arm in ('real_source','gan_source'):
        ax.hist(probabilities[(SEEDS[0],arm)][:360],
            bins=np.linspace(0,1,16),alpha=.4,density=True,label=arm)
    ax.set(xlabel='Estimated probability of target texture domain',ylabel='Density')
    ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(a.output/'real_neu_domain_predictions.png',dpi=190);plt.close(fig)
    manifest={'source':'NEU-DET genuine Kaggle steel-defect images',
        'source_zip_sha256':hashlib.sha256(a.neu_zip.read_bytes()).hexdigest(),
        'target':'three genuine scikit-image texture photographs: brick/grass/gravel',
        'source_train_available':1080,'source_train_used':120,'source_test':360,
        'target_train_tiles':120,'target_test_tiles':72,'target_train_rows':[0,1,2,3,4],
        'target_test_rows':[5,6,7],'seeds':list(SEEDS),
        'scope':'real-vs-synthetic source domain discrimination; not generator transfer',
        'limitations':['only three source photographs in texture target','correlated target patches',
            'fixed image features and logistic regression','single real source held-out split',
            'pretrained saved GANs are not retrained','not a general cross-domain augmentation benchmark']}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
