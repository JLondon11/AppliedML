"""Real-photo DCGAN diagnostic. These textures are NOT labelled defects.

Run: python run.py --steps 240 --output results
Real scikit-image brick/gravel/grass photographs, disjoint spatial columns.
A texture feature metric is reported; it is not FID or an anomaly benchmark.
"""
import argparse, csv, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from skimage import data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=20261106

def generate_patches(im, xlo, xhi, patch=32, stride=16):
    return np.stack([im[y:y+patch,x:x+patch] for y in range(0,im.shape[0]-patch+1,stride)
                     for x in range(xlo,xhi-patch+1,stride)]).astype('float32')/127.5-1

class Generator(nn.Module):
    def __init__(self,z=32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(z,128*4*4),nn.Unflatten(1,(128,4,4)),
        nn.ConvTranspose2d(128,64,4,2,1),nn.BatchNorm2d(64),nn.ReLU(),
        nn.ConvTranspose2d(64,32,4,2,1),nn.BatchNorm2d(32),nn.ReLU(),
        nn.ConvTranspose2d(32,1,4,2,1),nn.Tanh())
    def forward(self,z):return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,32,4,2,1),nn.LeakyReLU(0.2),
        nn.Conv2d(32,64,4,2,1),nn.LeakyReLU(0.2),
        nn.Conv2d(64,128,4,2,1),nn.LeakyReLU(0.2),
        nn.Flatten(),nn.Linear(128*4*4,1))
    def forward(self,x):return self.net(x)

def texture_features(arr):
    """Intensity mean/std, gradient magnitudes, lag-1 pixel correlation."""
    x=arr.reshape(-1,32,32).astype('float64')
    mean=x.mean((1,2)); std=x.std((1,2))
    dx=np.abs(np.diff(x,axis=2)).mean((1,2))
    dy=np.abs(np.diff(x,axis=1)).mean((1,2))
    lag=np.mean((x[:,:,:-1]-mean[:,None,None])*(x[:,:,1:]-mean[:,None,None]),(1,2))/(std**2+1e-8)
    return np.stack([mean,std,dx,dy,lag],axis=1)

def dist(a,b,scale):
    """Euclidean distance of standardized mean texture features, not FID."""
    return float(np.linalg.norm((texture_features(a).mean(0)-texture_features(b).mean(0))/scale))

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=240)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    a=p.parse_args();out=a.output;fig=out/'figures';fig.mkdir(parents=True,exist_ok=True)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
    imgs={name:getattr(data,name)() for name in ('brick','gravel','grass')}
    tr={k:generate_patches(im,0,384) for k,im in imgs.items()}
    te={k:generate_patches(im,384,512) for k,im in imgs.items()}
    sources={k:hashlib.sha256(im.tobytes()).hexdigest() for k,im in imgs.items()}
    z_dim=32;G=Generator(z_dim);D=Discriminator()
    optg=torch.optim.Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
    optd=torch.optim.Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))
    bce=nn.BCEWithLogitsLoss()
    perclass=[];train_log=[];allgenerated=[]
    for classname in imgs:
        torch.manual_seed(SEED);G=Generator(z_dim);D=Discriminator()
        optg=torch.optim.Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
        optd=torch.optim.Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))
        x=torch.from_numpy(tr[classname][:,None])
        for step in range(1,a.steps+1):
            idx=torch.randint(len(x),(32,));real=x[idx]
            fake=G(torch.randn(32,z_dim))
            D.train();G.train()
            ld=(bce(D(real),torch.ones(32,1))+bce(D(fake.detach()),torch.zeros(32,1)))/2
            optd.zero_grad();ld.backward();optd.step()
            fake=G(torch.randn(32,z_dim))
            lg=bce(D(fake),torch.ones(32,1))
            optg.zero_grad();lg.backward();optg.step()
            train_log.append((classname,step,float(ld.detach()),float(lg.detach())))
        G.eval()
        with torch.no_grad():gen=G(torch.randn(100,z_dim)).numpy()[:,0]
        np.save(out/f'generated_{classname}.npy',gen)
        allgenerated.append((classname,gen))
        heldout=te[classname];ref=texture_features(tr[classname])
        scale=np.maximum(ref.std(0),0.05)
        baseline=np.broadcast_to(np.array([tr[classname][i].mean() for i in range(100)])[:,None,None],(100,32,32))
        rng=np.random.default_rng(SEED)
        take=rng.choice(len(tr[classname]),100,replace=False)
        for method,images in [('GAN',gen),('constant_mean_baseline',baseline),
                              ('real_train_patches_reference',tr[classname][take])]:
            perclass.append({'texture':classname,'method':method,
                'n_train_patches':len(tr[classname]),'n_heldout_patches':len(heldout),
                'n_evaluation_samples':100,
                'standardized_mean_feature_distance':round(dist(images,heldout,scale),6)})
        torch.save({'model':G.state_dict(),'steps':a.steps,'seed':SEED,'class':classname},
                   out/f'generator_{classname}.pt')
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(perclass[0]));w.writeheader();w.writerows(perclass)
    with (out/'training_log.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['texture','step','loss_D','loss_G']);w.writerows(train_log)
    fig1,axes=plt.subplots(3,8,figsize=(12,5),constrained_layout=True)
    for i,(name,gen) in enumerate(allgenerated):
        for j in range(8):
            axes[i,j].imshow((gen[j]+1)/2,cmap='gray',vmin=0,vmax=1)
            axes[i,j].axis('off')
            if j==0:axes[i,j].set_ylabel(name)
    fig1.savefig(fig/'gan_texture_samples.png',dpi=180);plt.close(fig1)
    fig2,axes=plt.subplots(1,3,figsize=(11,3.1),constrained_layout=True)
    for ax,classname in zip(axes,imgs):
        lg=np.array([(step,ld,lv) for cls,step,ld,lv in train_log if cls==classname])
        ax.plot(lg[:,0],lg[:,1],lw=0.9,label='D loss')
        ax.plot(lg[:,0],lg[:,2],lw=0.9,label='G loss')
        ax.set(xlabel='Training step',ylabel='BCE loss',title=classname)
        ax.legend(frameon=False,fontsize=7)
    fig2.savefig(fig/'gan_training_losses.png',dpi=180);plt.close(fig2)
    fig3,ax=plt.subplots(figsize=(9,3.8),constrained_layout=True)
    methods=['GAN','constant_mean_baseline','real_train_patches_reference']
    group=np.arange(3)
    for j,method in enumerate(methods):
        ys=[next(r['standardized_mean_feature_distance'] for r in perclass
                 if r['texture']==cl and r['method']==method) for cl in imgs]
        ax.bar(group+(j-1)*0.24,ys,width=0.23,label=method.replace('_',' '))
    ax.set_xticks(group,list(imgs))
    ax.set_ylabel('Standardized mean texture-feature distance (lower better)')
    ax.legend(frameon=False,fontsize=8)
    fig3.savefig(fig/'gan_texture_feature_comparison.png',dpi=180);plt.close(fig3)
    manifest={'source':'skimage.data brick, gravel, grass photographs',
      'source_sha256':sources,'train_region_columns':[0,383],
      'heldout_region_columns':[384,511],'patch_size':32,'stride':16,
      'seed':SEED,'steps_per_class':a.steps,
      'model':'DCGAN unconditional grayscale patch generator, separate model per source',
      'metrics':'standardized distance of mean intensity/gradient/correlation features',
      'not_claimed':['defect generation','FID','real anomaly detection','industrial benchmark'],
      'limitations':['one photo per texture; spatial dependence of patches',
                     'weak pixel-mean baseline; descriptive comparison only',
                     'no defect labels or defect-generation evaluation']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(perclass,indent=2))
if __name__=='__main__':main()
