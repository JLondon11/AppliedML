"""NEU steel-defect conditional GAN augmentation benchmark.

Data are NOT bundled. No real NEU results are claimed until a real-data run.
Run: python run.py --data-root /path/to/NEU-DET --output results
Supported input: flat JPG names prefixed with six NEU labels, or class folders.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ('crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches')
SEED = 20261106
SIZE = 64
Z = 32

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize_label(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())

def discover_images(root):
    extensions = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in extensions]
    records = []
    for p in sorted(files):
        # Parent directory takes precedence; otherwise use filename prefix.
        candidates = (p.parent.name, p.stem)
        matched = []
        for raw in candidates:
            cleaned = normalize_label(raw)
            valid = [i for i, label in enumerate(CLASSES) if cleaned == normalize_label(label)
                     or cleaned.startswith(normalize_label(label))]
            if valid:
                matched = valid
                break
        if len(matched) != 1:
            continue
        records.append((p, matched[0]))
    if not records:
        raise ValueError('No recognized NEU defect class photographs found; check class folders/filenames.')
    return records

def prepare_dataset(root, output, smoke_test=False):
    records = discover_images(root)
    # Hash raw pixels rather than filenames so duplicate images across classes/splits are detected.
    data = []
    seen = {}
    for path, label in records:
        with Image.open(path) as img:
            if not smoke_test and img.size != (200, 200):
                raise ValueError(f'Expected NEU source dimensions 200x200: {path} is {img.size}')
            arr = np.asarray(img.convert('L').resize((SIZE,SIZE),Image.Resampling.BILINEAR), dtype=np.uint8)
        raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        decoded_sha = hashlib.sha256(arr.tobytes()).hexdigest()
        if decoded_sha in seen:
            raise ValueError(f'Duplicate decoded image data {path} and {seen[decoded_sha]}; remove duplicate before splitting.')
        seen[decoded_sha] = path
        data.append((path,label,raw_sha,decoded_sha,arr))
    labels = np.array([r[1] for r in data]); counts=Counter(labels)
    if smoke_test:
        if len(counts) != 6 or min(counts.values()) < 8:
            raise ValueError('Smoke fixture must contain at least eight unique images per class')
        ids = np.arange(len(data))
        trainids, rem = train_test_split(ids,test_size=0.40,stratify=labels,random_state=SEED)
        validids,testids = train_test_split(rem,test_size=0.50,stratify=labels[rem],random_state=SEED)
        splits = {'train':trainids,'validation':validids,'test':testids}
        split_strategy = 'smoke-test stratified individual image split; NOT NEU evidence'
    else:
        if len(data) != 1800 or any(counts.get(k,0) != 300 for k in range(6)):
            raise ValueError(f'Official NEU benchmark must have 300 unique images per class (1800 total): {dict(counts)}')
        # Consecutive class IDs are split into 30-image blocks to reduce near-neighbor leakage.
        class_groups = defaultdict(dict)
        for i,(path,label,*_) in enumerate(data):
            match = re.search(r'(\d+)$',path.stem)
            if not match:raise ValueError(f'Cannot construct blocked NEU split: missing numeric ID in {path.name}')
            class_groups[label][int(match.group(1))] = i
        index_splits = defaultdict(list)
        for label in range(6):
            identifiers = class_groups[label]
            if len(identifiers) != 300:
                raise ValueError('Duplicate class-wise numerical IDs in NEU filenames')
            base=min(identifiers)
            if sorted(identifiers) != list(range(base,base+300)):
                raise ValueError('NEU source image IDs are not 300 consecutive numbers within a class')
            blocks=list(range(10))
            random.Random(SEED+label).shuffle(blocks)
            assignments={'train':set(blocks[:6]),'validation':set(blocks[6:8]),'test':set(blocks[8:])}
            for n,i in identifiers.items():
                block=(n-base)//30
                for split,chosen in assignments.items():
                    if block in chosen:index_splits[split].append(i);break
        splits={k:np.asarray(index_splits[k],dtype=int) for k in ('train','validation','test')}
        assert [len(splits[k]) for k in ('train','validation','test')]==[1080,360,360]
        split_strategy='class-wise numbered 30-image blocked split: six train/two validation/two test blocks per class'
    split_of={int(j):s for s,ids_ in splits.items() for j in ids_}
    with (output/'dataset_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','filename','class','split','file_sha256','decoded_sha256'])
        for i,(p,label,sha,decoded,arr) in enumerate(data):
            w.writerow([i,str(p.relative_to(root)),CLASSES[label],split_of[i],sha,decoded])
    assert len(split_of)==len(data)
    return data, splits, split_strategy

class Photographs(Dataset):
    def __init__(self, data, indices): self.data=data;self.indices=list(indices)
    def __len__(self):return len(self.indices)
    def __getitem__(self,ix):
        _,label,_,_,array=self.data[self.indices[ix]]
        return torch.from_numpy(array.astype(np.float32)[None]/127.5-1),int(label)

class CondGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(6,16)
        self.net=nn.Sequential(nn.Linear(Z+16,256*4*4), nn.Unflatten(1,(256,4,4)),
            nn.ConvTranspose2d(256,128,4,2,1),nn.BatchNorm2d(128),nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1),nn.BatchNorm2d(64),nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1),nn.BatchNorm2d(32),nn.ReLU(True),
            nn.ConvTranspose2d(32,16,4,2,1),nn.BatchNorm2d(16),nn.ReLU(True),
            nn.Conv2d(16,1,3,padding=1),nn.Tanh())
    def forward(self,z,y):return self.net(torch.cat([z,self.emb(y)],dim=1))

class CondDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(6,SIZE*SIZE)
        self.net=nn.Sequential(nn.Conv2d(2,32,4,2,1),nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,4,2,1),nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1),nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,4,2,1),nn.LeakyReLU(0.2),nn.Flatten(),nn.Linear(256*4*4,1))
    def forward(self,x,y):return self.net(torch.cat([x,self.emb(y).reshape(-1,1,SIZE,SIZE)],dim=1))

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
             nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
             nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,6))
    def forward(self,x):return self.layers(x)

def train_gan(data,splits,steps,batch,device,output):
    loader=DataLoader(Photographs(data,splits['train']),batch_size=batch,shuffle=True,drop_last=True)
    if not len(loader):raise ValueError('GAN batch size exceeds training set')
    gen=CondGenerator().to(device);disc=CondDiscriminator().to(device)
    og=torch.optim.Adam(gen.parameters(),lr=0.0002,betas=(0.5,0.999))
    od=torch.optim.Adam(disc.parameters(),lr=0.0002,betas=(0.5,0.999))
    bce=nn.BCEWithLogitsLoss(); iterator=iter(loader); trace=[]
    for step in range(steps):
        try:x,y=next(iterator)
        except StopIteration:iterator=iter(loader);x,y=next(iterator)
        x,y=x.to(device),y.to(device);size=len(y)
        fake=gen(torch.randn(size,Z,device=device),y)
        lossd=(bce(disc(x,y),torch.ones(size,1,device=device))+
               bce(disc(fake.detach(),y),torch.zeros(size,1,device=device)))/2
        od.zero_grad();lossd.backward();od.step()
        fake=gen(torch.randn(size,Z,device=device),y)
        lossg=bce(disc(fake,y),torch.ones(size,1,device=device))
        og.zero_grad();lossg.backward();og.step()
        trace.append([step+1,round(float(lossd.detach()),6),round(float(lossg.detach()),6)])
    with (output/'gan_training.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['step','discriminator_BCE','generator_BCE']);w.writerows(trace)
    torch.save({'model':gen.state_dict(),'gan_steps':steps,'seed':SEED},output/'generator.pt')
    gen.eval()
    return gen,trace

def make_generated(gen,n_per_class,device):
    # Generate EXACT class-balanced samples; generated data is used only for classifier training.
    out_x=[];out_y=[]
    with torch.no_grad():
        for c in range(6):
            for j in range(0,n_per_class,64):
                n=min(64,n_per_class-j);y=torch.full((n,),c,device=device,dtype=torch.long)
                x=gen(torch.randn(n,Z,device=device),y).cpu()
                out_x.append(x);out_y.extend([c]*n)
    return torch.cat(out_x),torch.tensor(out_y,dtype=torch.long)

class Combined(Dataset):
    def __init__(self,real,generated=None):self.real=real;self.gen=generated
    def __len__(self):return len(self.real)+(0 if self.gen is None else len(self.gen[0]))
    def __getitem__(self,i):
        if i<len(self.real):return self.real[i]
        j=i-len(self.real);return self.gen[0][j],int(self.gen[1][j])

def evaluate(model,loader,device):
    model.eval();truth=[];prob=[]
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device));probs=torch.softmax(logits,dim=1).cpu().numpy()
            truth.extend(y.numpy().tolist());prob.extend(probs.tolist())
    pred=np.asarray(prob).argmax(1)
    return np.asarray(truth),pred,np.asarray(prob)

def train_classifier(data,splits,extra,epochs,batch,device,output,tag):
    # Reinitialize identically across arms; GAN images only augment training.
    torch.manual_seed(SEED+10)
    model=Classifier().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    trainloader=DataLoader(Combined(Photographs(data,splits['train']),extra),batch_size=batch,shuffle=True)
    valloader=DataLoader(Photographs(data,splits['validation']),batch_size=batch,shuffle=False)
    trace=[];best=-1;best_epoch=None;best_weights=None
    for epoch in range(epochs):
        model.train();losses=[]
        for x,y in trainloader:
            x,y=x.to(device),y.to(device)
            loss=nn.functional.cross_entropy(model(x),y)
            optimizer.zero_grad();loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        true,pred,_=evaluate(model,valloader,device)
        val_bal=float(balanced_accuracy_score(true,pred))
        trace.append([epoch+1,float(np.mean(losses)),val_bal])
        if val_bal>best:
            best=val_bal;best_epoch=epoch+1
            best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    torch.save({'model':best_weights,'selected_epoch':best_epoch,'val_balanced_accuracy':best,
                'seed':SEED+10},output/f'classifier_{tag}.pt')
    with (output/f'classifier_training_{tag}.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','train_cross_entropy','validation_balanced_accuracy']);w.writerows(trace)
    return model,best_epoch,best,trace

def report_results(data,splits,models,device,output):
    loader=DataLoader(Photographs(data,splits['test']),batch_size=64,shuffle=False)
    results=[]; rows=[];matrices=[]
    for tag,(model,selected_epoch,val,trace) in models.items():
        true,pred,probs=evaluate(model,loader,device)
        assert len(true)==len(splits['test'])
        result={'arm':tag,'selected_epoch':selected_epoch,'validation_balanced_accuracy':val,
                'test_accuracy':float(accuracy_score(true,pred)),
                'test_balanced_accuracy':float(balanced_accuracy_score(true,pred)),
                'test_macro_F1':float(f1_score(true,pred,average='macro',zero_division=0))}
        results.append(result)
        matrices.append((tag,confusion_matrix(true,pred,labels=list(range(6)))))
        for local_idx,image_idx in enumerate(splits['test']):
            rows.append([tag,int(image_idx),CLASSES[true[local_idx]],CLASSES[pred[local_idx]],
                        *[float(t) for t in probs[local_idx]]])
    with (output/'test_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','dataset_index','true_class','predicted_class',
                                   *['prob_'+s for s in CLASSES]]);w.writerows(rows)
    with (output/'benchmark_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,(tag,mat) in zip(axs,matrices):
        im=ax.imshow(mat,cmap='Blues');ax.set_title(tag);ax.set_xlabel('Predicted');ax.set_ylabel('True')
        ax.set_xticks(np.arange(6),CLASSES,rotation=60,ha='right',fontsize=8)
        ax.set_yticks(np.arange(6),CLASSES,fontsize=8)
        for (r,c),val in np.ndenumerate(mat):ax.text(c,r,str(val),ha='center',va='center',fontsize=8)
    fig.savefig(output/'figures'/'test_confusion_matrices.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.5))
    for tag,(_,_,_,trace) in models.items():
        ax.plot([r[0] for r in trace],[r[2] for r in trace],label=tag)
    ax.set(xlabel='Training epoch',ylabel='Validation balanced accuracy',ylim=(0,1))
    ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(output/'figures'/'validation_balanced_accuracy.png',dpi=180);plt.close(fig)
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--gan-steps',type=int,default=1500)
    p.add_argument('--classifier-epochs',type=int,default=16)
    p.add_argument('--generated-per-class',type=int,default=150)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--smoke-test',action='store_true',help='Test-only fixture; do not claim real NEU measurements')
    a=p.parse_args();seed_all(SEED)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'figures').mkdir(exist_ok=True)
    data,splits,split_strategy=prepare_dataset(a.data_root,a.output,a.smoke_test)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan,history=train_gan(data,splits,a.gan_steps,a.batch_size,device,a.output)
    generated=make_generated(gan,a.generated_per_class,device)
    torch.save({'images':generated[0],'labels':generated[1]},a.output/'generated_training_samples.pt')
    baseline=train_classifier(data,splits,None,a.classifier_epochs,a.batch_size,device,a.output,'real_only')
    augmented=train_classifier(data,splits,generated,a.classifier_epochs,a.batch_size,device,a.output,'gan_augmented')
    results=report_results(data,splits,{'real_only':baseline,'gan_augmented':augmented},device,a.output)
    nview=min(6,a.generated_per_class)
    fig,axes=plt.subplots(6,nview,figsize=(1.5*nview,9),constrained_layout=True,squeeze=False)
    for i in range(6):
        gen=generated[0][generated[1]==i][:nview].numpy()[:,0]
        for j in range(nview):
            axes[i,j].imshow((gen[j]+1)/2,cmap='gray',vmin=0,vmax=1);axes[i,j].axis('off')
            if j==0:axes[i,j].set_ylabel(CLASSES[i])
    fig.savefig(a.output/'figures'/'conditional_gan_samples.png',dpi=160);plt.close(fig)
    manifest={'data_root':str(a.data_root.resolve()),'classes':CLASSES,
      'n_images':len(data),'split_sizes':{k:len(v) for k,v in splits.items()},
      'seed':SEED,'gan_steps':a.gan_steps,'classifier_epochs':a.classifier_epochs,
      'gan_samples_per_class':a.generated_per_class,'batch_size':a.batch_size,
      'smoke_fixture_mode':bool(a.smoke_test), 'split_strategy':split_strategy,
      'real_NEU_results_claimed':False, 'data_structure_passed':not a.smoke_test,
      'critical_limitations':['Blocked split reduces but cannot eliminate near-duplicate leakage',
        'GAN evaluation based on downstream classification and sample gallery; no FID/KID',
        'Results cannot be claimed without verifying dataset lineage and independent test'],
      'comparisons':results}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
