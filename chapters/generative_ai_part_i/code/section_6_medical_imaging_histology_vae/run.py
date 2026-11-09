"""Real immunohistochemistry tissue-patch VAE. Not patient-level medical validation.

Real RGB microscopy image: scikit-image data.immunohistochemistry().
Spatially disjoint horizontal row regions for train/validation/test.
"""
import csv, json, random, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from skimage import data,metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=20261109
class VAE(nn.Module):
    def __init__(self,z=24):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.ReLU(),nn.Conv2d(16,32,4,2,1),nn.ReLU(),nn.Conv2d(32,64,4,2,1),nn.ReLU(),nn.Flatten())
        self.mu=nn.Linear(64*4*4,z);self.lv=nn.Linear(64*4*4,z)
        self.fc=nn.Linear(z,64*4*4)
        self.dec=nn.Sequential(nn.Unflatten(1,(64,4,4)),nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,3,4,2,1),nn.Sigmoid())
    def encode(self,x):
        h=self.enc(x);return self.mu(h),self.lv(h).clamp(-8,8)
    def decode(self,z):return self.dec(self.fc(z))
    def forward(self,x):
        mu,lv=self.encode(x);z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
        return self.decode(z),mu,lv

def make_tiles(image,first,last):
    result=[];positions=[]
    for row in range(first,last-31,16):
        for col in range(0,512-31,16):
            result.append(np.moveaxis(image[row:row+32,col:col+32],-1,0))
            positions.append((row,col))
    return np.stack(result).astype('float32')/255, positions

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=p.parse_args();out=args.output;fig=out/'figures';fig.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    real=data.immunohistochemistry()
    assert real.shape==(512,512,3)
    sets={};pos={}
    for name,start,stop in [('train',0,320),('validation',320,384),('test',384,512)]:
        sets[name],pos[name]=make_tiles(real,start,stop)
    model=VAE();opt=torch.optim.Adam(model.parameters(),lr=.001)
    tx=torch.from_numpy(sets['train']);vx=torch.from_numpy(sets['validation']);test=torch.from_numpy(sets['test'])
    best=1e9;state=None;selected=None;trace=[]
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(len(tx));losses=[]
        for inds in order.split(64):
            x=tx[inds];recon,mu,lv=model(x)
            mse=(recon-x).square().mean();kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(-1)).mean()
            loss=mse+.0001*kl;opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            preds=[]
            for batch in vx.split(64):
                mu,_=model.encode(batch);preds.append(model.decode(mu))
            val=float((torch.cat(preds)-vx).square().mean())
        trace.append((epoch,float(np.mean(losses)),val))
        print('epoch',epoch,'train',round(np.mean(losses),5),'validation',round(val,5),flush=True)
        if val<best:best=val;selected=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state);model.eval()
    with torch.no_grad():
        reconstructed=torch.cat([model.decode(model.encode(x)[0]) for x in test.split(64)])
        z=torch.randn(128,24);generated=model.decode(z)
    reference=np.moveaxis(test.numpy(),1,-1);recons=np.moveaxis(reconstructed.numpy(),1,-1)
    synth=np.moveaxis(generated.numpy(),1,-1)
    train_flat=sets['train'].reshape(len(tx),-1)
    test_flat=test.numpy().reshape(len(test),-1)
    closest=[]
    for i in range(0,len(test),16):
        a=test_flat[i:i+16];d=np.mean((a[:,None,:]-train_flat[None,:,:])**2,axis=2)
        closest.append(np.min(d,axis=1))
    nearest=np.concatenate(closest)
    sample_nn=[]
    gen_flat=generated.numpy().reshape(len(generated),-1)
    for i in range(0,len(generated),16):
        a=gen_flat[i:i+16];d=np.mean((a[:,None,:]-train_flat[None,:,:])**2,axis=2)
        sample_nn.extend(np.min(d,axis=1).tolist())
    records=[]
    for i,(ref,rec) in enumerate(zip(reference,recons)):
        records.append(dict(test_index=i,row=pos['test'][i][0],col=pos['test'][i][1],
            recon_PSNR_dB=float(metrics.peak_signal_noise_ratio(ref,rec,data_range=1)),
            recon_SSIM=float(metrics.structural_similarity(ref,rec,channel_axis=-1,data_range=1)),
            recon_MSE=float(np.mean((ref-rec)**2)),nearest_training_patch_MSE=float(nearest[i])))
    with (out/'heldout_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with (out/'prior_samples.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','nearest_training_patch_MSE','mean_RGB','std_RGB'])
        w.writerows((i,float(sample_nn[i]),float(im.mean()),float(im.std())) for i,im in enumerate(synth))
    with (out/'training.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','train_vae_objective','validation_reconstruction_MSE']);w.writerows(trace)
    m={'dataset':'scikit-image immunohistochemistry tissue microscopy 512x512 RGB','source_sha256':hashlib.sha256(real.tobytes()).hexdigest(),
      'source_images':1,'spatial_partitions':{'train_rows':[0,319],'validation_rows':[320,383],'test_rows':[384,511]},
      'n_train_patches':len(tx),'n_val_patches':len(vx),'n_test_patches':len(test),'n_prior_samples':len(synth),
      'seed':SEED,'epochs':args.epochs,'selected_epoch':selected,
      'test_reconstruction_PSNR_dB':float(np.mean([r['recon_PSNR_dB'] for r in records])),
      'test_reconstruction_SSIM':float(np.mean([r['recon_SSIM'] for r in records])),
      'test_reconstruction_MSE':float(np.mean([r['recon_MSE'] for r in records])),
      'test_nearest_training_patch_MSE':float(nearest.mean()),
      'prior_nearest_training_patch_MSE':float(np.mean(sample_nn)),
      'limitations':['one tissue source image; not patient independent','patch overlap within each partition','prior sample quality not judged by a pathologist','reconstruction scores are not unconditional-generation quality','no pathology labels or downstream clinical utility','no GAN/diffusion benchmark']}
    (out/'metrics.json').write_text(json.dumps(m,indent=2))
    torch.save({'model':state,'seed':SEED,'selected_epoch':selected,'metrics':m},out/'tissue_vae.pt')
    fig1,axes=plt.subplots(3,8,figsize=(13,5.8),constrained_layout=True)
    for k,i in enumerate(np.linspace(0,len(test)-1,8,dtype=int)):
        axes[0,k].imshow(reference[i]);axes[1,k].imshow(recons[i]);axes[2,k].imshow(synth[k])
        for r in range(3):axes[r,k].axis('off')
    for r,label in enumerate(['Held-out real tissue','VAE reconstruction','Unconditional VAE sample']):axes[r,0].set_title(label,fontsize=9,loc='left')
    fig1.savefig(fig/'tissue_real_recon_generated.png',dpi=170);plt.close(fig1)
    fig2,ax=plt.subplots(figsize=(7,4))
    ax.plot([t[0] for t in trace],[t[1] for t in trace],label='Training VAE objective')
    ax.plot([t[0] for t in trace],[t[2] for t in trace],label='Validation reconstruction MSE')
    ax.set(xlabel='Epoch',ylabel='Loss (different objective definitions)');ax.legend(frameon=False);fig2.tight_layout()
    fig2.savefig(fig/'tissue_vae_training.png',dpi=180);plt.close(fig2)
    print(json.dumps(m,indent=2),flush=True)
if __name__=='__main__':main()
