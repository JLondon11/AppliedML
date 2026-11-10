"""Chapter 9 Figure 8 candidate: measured image/label-word contrastive alignment.

Images and their human annotated class labels come from sklearn's real UCI digits.
The text side is the *actual class name*, not natural captions or a foundation model.
A class-supervised cross-entropy contrastive objective learns aligned image/text
embeddings. Train/validation/test partitions are disjoint.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

NAMES=('zero','one','two','three','four','five','six','seven','eight','nine')
ALPHABET='abcdefghijklmnopqrstuvwxyz'
TOKENS=torch.tensor([[ALPHABET.index(c)+1 for c in word]+[0]*(len(max(NAMES,key=len))-len(word)) for word in NAMES],dtype=torch.long)
PALETTE=['#24465B','#B26E44','#4B7F78','#846D97','#B68D3C','#5B718C','#AC6977','#7D8751','#6D8590','#9B7668']

class ImageLabelAligner(nn.Module):
    def __init__(self,dim=32):
        super().__init__()
        self.image=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,dim))
        self.characters=nn.Embedding(27,24,padding_idx=0)
        self.words=nn.Sequential(nn.Linear(24,64),nn.GELU(),nn.Linear(64,dim))
        self.logit_scale=nn.Parameter(torch.tensor(2.3))

    def encode_text(self,tokens):
        mask=(tokens!=0).float().unsqueeze(-1)
        pooled=(self.characters(tokens)*mask).sum(1)/mask.sum(1).clamp_min(1)
        return F.normalize(self.words(pooled),dim=-1)

    def forward(self,images,tokens=TOKENS):
        i=F.normalize(self.image(images),dim=-1)
        t=self.encode_text(tokens.to(images.device))
        return torch.exp(self.logit_scale.clamp(max=4))*i@t.T,i,t


def run(out:Path,epochs=60,seed=53):
    if epochs<1:raise ValueError('epochs must be positive')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    d=load_digits();X=torch.tensor(d.data.astype('float32')/16);y=d.target
    idx=np.arange(len(y))
    trva,te=train_test_split(idx,test_size=270,stratify=y,random_state=seed)
    tr,va=train_test_split(trva,test_size=270,stratify=y[trva],random_state=seed)
    assert (len(tr),len(va),len(te))==(1257,270,270)
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    model=ImageLabelAligner();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=torch.Generator().manual_seed(seed+100)
    best=-1;best_epoch=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0
        for batch in torch.randperm(len(tr),generator=rng).split(128):
            opt.zero_grad()
            logits,_,_=model(X[tr[batch]])
            loss=F.cross_entropy(logits,torch.tensor(y[tr[batch]],dtype=torch.long))
            loss.backward();opt.step()
            loss_sum+=loss.item()*len(batch)
        model.eval()
        with torch.no_grad():
            v_logits,_,_=model(X[va]);vacc=float((v_logits.argmax(1)==torch.tensor(y[va])).float().mean())
        history.append({'epoch':epoch,'train_contrastive_ce':loss_sum/len(tr),'validation_top1':vacc})
        if vacc>best:
            best=vacc;best_epoch=epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'seed':seed},out/'aligner_best.pt')
    state=torch.load(out/'aligner_best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['model']);model.eval()
    with torch.no_grad():
        logits,image_features,text_features=model(X[te])
        probs=logits.softmax(-1).numpy()
        scores=logits.numpy()
        p=logits.argmax(1).numpy()
    accuracy=float((p==y[te]).mean());cm=confusion_matrix(y[te],p,labels=range(10))
    mean_matrix=np.stack([scores[y[te]==i].mean(axis=0) for i in range(10)])
    embeddings=PCA(n_components=2,random_state=seed).fit_transform(image_features.numpy())
    fig=plt.figure(figsize=(12.2,7.2),facecolor='white')
    gs=fig.add_gridspec(2,3,height_ratios=[1,1],width_ratios=[1,1.15,1.2],wspace=.38,hspace=.43)
    ax1=fig.add_subplot(gs[0,0]);ax2=fig.add_subplot(gs[0,1]);ax3=fig.add_subplot(gs[0,2]);ax4=fig.add_subplot(gs[1,:])
    example=int(np.flatnonzero(p==y[te])[0])
    ax1.imshow(X[te[example]].reshape(8,8),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    ax1.set_title('Actual held-out digit',fontsize=10);ax1.axis('off')
    im=ax2.imshow(mean_matrix,cmap='viridis',aspect='auto')
    ax2.set(title='Observed image–label score',xlabel='Text class label',ylabel='Real image class')
    ax2.set_xticks(range(10),NAMES,rotation=65,fontsize=7);ax2.set_yticks(range(10),NAMES,fontsize=7)
    fig.colorbar(im,ax=ax2,shrink=.75,label='Mean trained score')
    ax3.imshow(cm,cmap='Blues',aspect='equal')
    ax3.set(title='Independent test predictions',xlabel='Predicted digit',ylabel='Real digit')
    ax3.set_xticks(range(10));ax3.set_yticks(range(10))
    for i in range(10):
        ax4.scatter(embeddings[y[te]==i,0],embeddings[y[te]==i,1],s=16,alpha=.76,
                    label=NAMES[i],color=PALETTE[i],linewidths=0)
    ax4.set(title='PCA of actual held-out image embeddings',xlabel='Image embedding PC1',ylabel='Image embedding PC2')
    ax4.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=10,frameon=False,fontsize=7)
    fig.suptitle('Measured image-to-label-word alignment on real handwritten-digit images',fontsize=13,y=.99)
    fig.savefig(out/'Figure_008_Multimodal_foundation_model_fusion_architecture.png',dpi=210,bbox_inches='tight',facecolor='white')
    fig.savefig(out/'Figure_008_Multimodal_foundation_model_fusion_architecture.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    with (out/'test_predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset_index','real_digit','predicted_label','max_class_prob'])
        for j,index in enumerate(te):writer.writerow([int(index),int(y[index]),int(p[j]),float(probs[j].max())])
    with (out/'training_history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    metrics={
        'dataset_name':'scikit-learn UCI real handwritten digits with observed digit class labels',
        'dataset_source_url':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html',
        'split_counts':{'train':len(tr),'validation':len(va),'test':len(te)},'split_disjoint':True,
        'seed':seed,'epochs':epochs,'best_validation_epoch':best_epoch,'validation_top1':best,
        'independent_test_top1':accuracy,'correct_test_count':int((p==y[te]).sum()),
        'confusion_matrix':cm.tolist(),'image_text_mean_score':mean_matrix.tolist(),
        'text_labels_from_real_annotations':list(NAMES),
        'text_modality_type':'canonical human digit category names, not natural captions',
        'model':'32-dimensional learned image encoder and character-word encoder; supervised class-contrastive logits',
        'limitations':'Real images and observed class labels, but not pretrained CLIP, not human natural-language captions, not vision-language-audio foundation fusion. Experimental candidate only.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('confusion_matrix','image_text_mean_score')},indent=2))
    return metrics

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=60);a=ap.parse_args();run(a.out,a.epochs)
