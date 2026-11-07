"""Three-seed NEU-DET replication: real-only, flips, and conditional-GAN augmentation."""
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import prepare_dataset,Classifier,CondGenerator,CondDiscriminator,CLASSES,Z,SEED

class BufferedImages(Dataset):
    def __init__(self,arrays,labels,augment=False):
        self.x=arrays;self.y=labels;self.augment=augment
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        x=self.x[i]
        if self.augment and torch.rand(()) < 0.5:x=x.flip(-1)
        return x,self.y[i]

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(3)

def load_splits(data,splits):
    x=torch.from_numpy(np.stack([r[-1] for r in data]).astype('float32')[:,None]/127.5-1)
    y=torch.tensor([r[1] for r in data],dtype=torch.long)
    return {s:(x[ids],y[ids]) for s,ids in splits.items()}

def train_gan(train,seed,steps,batch,output):
    seed_all(seed+201)
    g=CondGenerator();d=CondDiscriminator()
    og=torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999))
    od=torch.optim.Adam(d.parameters(),lr=.0002,betas=(.5,.999))
    bce=nn.BCEWithLogitsLoss();x,y=train;history=[]
    g.train();d.train()
    for step in range(steps):
        idx=torch.randint(len(y),(batch,));real=x[idx];cls=y[idx]
        fake=g(torch.randn(batch,Z),cls)
        ld=(bce(d(real,cls),torch.ones(batch,1))+bce(d(fake.detach(),cls),torch.zeros(batch,1)))/2
        od.zero_grad();ld.backward();od.step()
        fake=g(torch.randn(batch,Z),cls)
        lg=bce(d(fake,cls),torch.ones(batch,1))
        og.zero_grad();lg.backward();og.step()
        history.append((seed,step+1,float(ld.detach()),float(lg.detach())))
    g.eval()
    torch.save({'model':g.state_dict(),'seed':seed,'steps':steps},output/f'gan_seed{seed}.pt')
    return g,history

def evaluate(model,subset,batch):
    model.eval();ys=[];prob=[]
    with torch.no_grad():
        for x,y in DataLoader(BufferedImages(*subset),batch_size=batch):
            p=model(x).softmax(1).numpy()
            prob.extend(p.tolist());ys.extend(y.numpy().tolist())
    ys=np.array(ys);prob=np.array(prob)
    return ys,prob.argmax(1),prob

def train_classifier(train,valid,test,arm,seed,epochs,batch,output,generated=None):
    seed_all(seed+10)
    net=Classifier();opt=torch.optim.Adam(net.parameters(),lr=.001)
    if arm=='flip_augmentation':ds=BufferedImages(*train,augment=True)
    elif generated is not None:
        ds=BufferedImages(torch.cat([train[0],generated[0]]),torch.cat([train[1],generated[1]]))
    else:ds=BufferedImages(*train)
    best=-1;selected=0;weights=None;history=[]
    for epoch in range(epochs):
        net.train();losses=[]
        for x,y in DataLoader(ds,batch_size=batch,shuffle=True):
            loss=nn.functional.cross_entropy(net(x),y)
            opt.zero_grad();loss.backward();opt.step()
            losses.append(float(loss.detach()))
        truth,pred,_=evaluate(net,valid,batch)
        v=float(balanced_accuracy_score(truth,pred))
        history.append((seed,arm,epoch+1,float(np.mean(losses)),v))
        if v>best:
            best=v;selected=epoch+1
            weights={k:t.detach().cpu().clone() for k,t in net.state_dict().items()}
    net.load_state_dict(weights)
    truth,pred,prob=evaluate(net,test,batch)
    metrics={'seed':seed,'arm':arm,'selected_epoch':selected,'validation_balanced_accuracy':best,
      'test_accuracy':float(accuracy_score(truth,pred)),
      'test_balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
      'test_macro_F1':float(f1_score(truth,pred,average='macro',zero_division=0))}
    torch.save({'model':weights,'seed':seed,'arm':arm,'selected_epoch':selected,
                'validation_balanced_accuracy':best},output/f'classifier_{arm}_seed{seed}.pt')
    preds=[(seed,arm,i,int(y),int(p),*map(float,ps))
        for i,(y,p,ps) in enumerate(zip(truth,pred,prob))]
    return metrics,history,preds,confusion_matrix(truth,pred,labels=list(range(6)))

def features(x):
    x=np.asarray(x,dtype='float64')
    return np.stack([x.mean((1,2)),x.std((1,2)),
        np.abs(np.diff(x,axis=2)).mean((1,2)),
        np.abs(np.diff(x,axis=1)).mean((1,2))],axis=1)

def generative_proxies(train,gen,seed):
    x,y=train;gx,gy=gen;out=[]
    for c in range(6):
        r=features(x[y==c][:,0].numpy());s=features(gx[gy==c][:,0].numpy())
        dist=float(np.linalg.norm((r.mean(0)-s.mean(0))/np.maximum(r.std(0),.03)))
        diversity=float(np.mean(s.std(0)/(r.std(0)+1e-6)))
        out.append((seed,CLASSES[c],dist,diversity))
    return out

def savecsv(path,cols,rows):
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(cols);writer.writerows(rows)

def gallery(generated,seed,path):
    x,y=generated;fig,axes=plt.subplots(6,6,figsize=(9,9),constrained_layout=True)
    for c in range(6):
        images=x[y==c][:6].numpy()[:,0]
        for j in range(6):
            axes[c,j].imshow((images[j]+1)/2,cmap='gray',vmin=0,vmax=1)
            axes[c,j].axis('off')
            if j==0:axes[c,j].set_ylabel(CLASSES[c],fontsize=8)
    fig.savefig(path,dpi=170);plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results_multiseed')
    p.add_argument('--seeds',type=int,nargs='+',default=[20261106,20261107,20261108])
    p.add_argument('--gan-steps',type=int,default=300)
    p.add_argument('--classifier-epochs',type=int,default=12)
    p.add_argument('--generated-per-class',type=int,default=120)
    p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args();start=time.time();seed_all(SEED)
    args.output.mkdir(parents=True,exist_ok=True)
    figs=args.output/'figures';figs.mkdir(exist_ok=True)
    data,splits,description=prepare_dataset(args.data_root,args.output,False)
    subsets=load_splits(data,splits)
    results=[];histories=[];predictions=[];gan_losses=[];qualities=[];cms={}
    for seed in args.seeds:
        g,losses=train_gan(subsets['train'],seed,args.gan_steps,args.batch_size,args.output)
        gan_losses.extend(losses)
        with torch.no_grad():
            seed_all(seed+501);xs=[];ys=[]
            for c in range(6):
                for j in range(0,args.generated_per_class,32):
                    n=min(32,args.generated_per_class-j);labels=torch.full((n,),c,dtype=torch.long)
                    xs.append(g(torch.randn(n,Z),labels).detach());ys.extend([c]*n)
            generated=torch.cat(xs),torch.tensor(ys,dtype=torch.long)
        torch.save({'x':generated[0],'y':generated[1],'seed':seed},args.output/f'generated_seed{seed}.pt')
        gallery(generated,seed,figs/f'gan_samples_seed{seed}.png')
        qualities.extend(generative_proxies(subsets['train'],generated,seed))
        for arm in ('real_only','flip_augmentation','gan_augmented'):
            m,h,preds,cm=train_classifier(subsets['train'],subsets['validation'],subsets['test'],arm,seed,
                args.classifier_epochs,args.batch_size,args.output,
                generated if arm=='gan_augmented' else None)
            results.append(m);histories.extend(h);predictions.extend(preds);cms[(seed,arm)]=cm
        print('COMPLETED',seed,results[-3:],flush=True)
    savecsv(args.output/'benchmark_metrics.csv',list(results[0]),[[r[k] for k in results[0]] for r in results])
    savecsv(args.output/'classifier_training.csv',
        ['seed','arm','epoch','train_loss','validation_balanced_accuracy'],histories)
    savecsv(args.output/'gan_training.csv',['seed','step','D_loss','G_loss'],gan_losses)
    savecsv(args.output/'test_predictions.csv',
        ['seed','arm','test_row','true_class','pred_class',*['p_'+s for s in CLASSES]],predictions)
    savecsv(args.output/'gan_quality_proxy.csv',
        ['seed','class','mean_feature_distance','feature_diversity_ratio'],qualities)
    fig,ax=plt.subplots(figsize=(8,4.5))
    arms=('real_only','flip_augmentation','gan_augmented')
    for j,arm in enumerate(arms):
        vals=np.array([r['test_balanced_accuracy'] for r in results if r['arm']==arm])
        ax.scatter(j+np.linspace(-.075,.075,len(vals)),vals,s=45)
        ax.plot([j-.13,j+.13],[vals.mean()]*2,lw=2)
    ax.set(xticks=range(3),xticklabels=['Real only','Horizontal flip','GAN augmented'],
           ylabel='Held-out balanced accuracy',ylim=(0,1))
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(figs/'test_accuracy_replicates.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for i,arm in enumerate(arms):
        cm=sum(cms[(s,arm)] for s in args.seeds)
        axes[i].imshow(cm,cmap='Blues');axes[i].set_title(arm)
        axes[i].set_xticks(range(6),CLASSES,rotation=65,ha='right',fontsize=8)
        axes[i].set_yticks(range(6),CLASSES,fontsize=8);axes[i].set_xlabel('Predicted class')
        if i==0:axes[i].set_ylabel('True class')
        for (r,c),val in np.ndenumerate(cm):
            axes[i].text(c,r,str(val),fontsize=8,ha='center',va='center')
    fig.savefig(figs/'combined_test_confusion_matrices.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for arm in arms:
        tr=np.asarray([[h[2],h[4]] for h in histories if h[1]==arm])
        curve=[tr[tr[:,0]==epoch,1].mean() for epoch in range(1,args.classifier_epochs+1)]
        ax.plot(range(1,args.classifier_epochs+1),curve,label=arm)
    ax.legend(frameon=False)
    ax.set(xlabel='Epoch',ylabel='Mean validation balanced accuracy',ylim=(0,1))
    fig.tight_layout();fig.savefig(figs/'validation_curves.png',dpi=190);plt.close(fig)
    manifest={'dataset':'user-uploaded Kaggle NEU-DET','dataset_sha256':'2ccaff85d88567c52708c2f9dc4dec10cff8650749df2984f03b38716998460f',
        'split_strategy':description,'split_sizes':{k:len(v) for k,v in splits.items()},
        'seeds':args.seeds,'gan_steps_per_seed':args.gan_steps,'classifier_epochs':args.classifier_epochs,
        'generated_per_class':args.generated_per_class,
        'quality_proxy':'handcrafted pixel moments/gradients, not FID/KID',
        'evaluation':'untouched original test subset; validation checkpoint selection',
        'limitations':['Only three seeds','300 GAN steps may not converge',
           'No expert or trained-model sample fidelity validation',
           'Horizontal flips not sample-count matched to GAN arm',
           'One held-out partition; external generalization not established'],
        'runtime_sec':round(time.time()-start,1)}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',manifest['runtime_sec'],flush=True)
if __name__=='__main__':main()
