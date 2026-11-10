"""Real handwritten-digits distributed Transformer training benchmark.

Measured one-rank versus two-rank PyTorch CPU/Gloo DistributedDataParallel.
Does not establish tensor, pipeline, expert, GPU, or multi-node parallelism.
"""
from __future__ import annotations
import argparse,json,os,random,resource,socket,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TransformerDigits(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(8,16)
        self.position=nn.Parameter(torch.zeros(1,8,16))
        self.layer=nn.TransformerEncoderLayer(d_model=16,nhead=2,dim_feedforward=32,dropout=0.,batch_first=True)
        self.head=nn.Linear(16,10)
    def forward(self,x):
        h=self.layer(self.embed(x.reshape(-1,8,8))+self.position)
        return self.head(h.mean(dim=1))

class DigitRows(Dataset):
    def __init__(self,x,y):self.x=x;self.y=y
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return self.x[i],self.y[i]

def splits(seed=67):
    d=load_digits()
    X=torch.tensor(d.data.astype('float32')/16)
    y=torch.tensor(d.target.astype('int64'))
    idx=np.arange(len(y))
    trv,te=train_test_split(idx,test_size=270,random_state=seed,stratify=y.numpy())
    tr,va=train_test_split(trv,test_size=270,random_state=seed,stratify=y[trv].numpy())
    assert (len(tr),len(va),len(te))==(1257,270,270)
    assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    return X,y,tr,va,te

def rank_run(rank,world,port,epochs,seed,out):
    torch.set_num_threads(1)
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']=str(port)
    dist.init_process_group(backend='gloo',rank=rank,world_size=world)
    try:
        X,y,tr,va,te=splits(seed)
        torch.manual_seed(seed)
        net=TransformerDigits()
        model=DDP(net)
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        ds=DigitRows(X[tr],y[tr])
        sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=False)
        loader=DataLoader(ds,batch_size=64//world,sampler=sampler,drop_last=False,num_workers=0)
        history=[];processed=0
        start=time.perf_counter()
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            model.train()
            for images,labels in loader:
                opt.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(images),labels)
                loss.backward()
                opt.step()
                processed+=len(labels)
            if rank==0:
                model.eval()
                with torch.no_grad():
                    logits=net(X[va])
                    val_ce=float(F.cross_entropy(logits,y[va]))
                    val_acc=float((logits.argmax(1)==y[va]).float().mean())
                history.append(dict(epoch=epoch+1,validation_ce=val_ce,validation_acc=val_acc))
        dist.barrier()
        elapsed=time.perf_counter()-start
        stats=torch.tensor([float(processed),float(elapsed),float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)],dtype=torch.float64)
        gathered=[torch.zeros(3,dtype=torch.float64) for _ in range(world)]
        dist.all_gather(gathered,stats)
        if rank==0:
            model.eval()
            with torch.no_grad():
                test_logits=net(X[te])
                test_ce=float(F.cross_entropy(test_logits,y[te]))
                test_accuracy=float((test_logits.argmax(1)==y[te]).float().mean())
            total=int(sum(z[0] for z in gathered))
            wall=max(float(z[1]) for z in gathered)
            report={
                'world_size':world,'epochs':epochs,'seed':seed,'nominal_global_batch':64,
                'per_rank_batch':64//world,'model_parameters':sum(p.numel() for p in net.parameters()),
                'total_samples_processed_including_padding':total,
                'sampler_padding_duplicates_per_epoch':total//epochs-len(tr),
                'wall_seconds_max_rank':wall,'measured_processed_samples_per_sec':total/wall,
                'rank_wall_seconds':[float(z[1]) for z in gathered],
                'rank_peak_rss_kb':[float(z[2]) for z in gathered],
                'validation_history':history,
                'test_cross_entropy':test_ce,'test_accuracy':test_accuracy}
            (out/f'world_{world}.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        dist.destroy_process_group()

def unused_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]

def run(out,epochs=4,seed=67):
    if epochs<1:raise ValueError('epochs must be positive')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    for world in (1,2):
        mp.spawn(rank_run,args=(world,unused_port(),epochs,seed,out),nprocs=world,join=True)
    observations=[json.loads((out/f'world_{world}.json').read_text()) for world in (1,2)]
    assert [z['world_size'] for z in observations]==[1,2]
    assert all(np.isfinite(z['test_cross_entropy']) for z in observations)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    keys=[('measured_processed_samples_per_sec','Observed CPU processed samples/s'),
          ('wall_seconds_max_rank','Training wall-clock seconds'),
          ('test_accuracy','Independent test accuracy')]
    for ax,(key,title) in zip(axes,keys):
        ax.bar([1,2],[z[key] for z in observations],color=['#294C60','#B26E3B'],width=.55)
        ax.set(xticks=[1,2],xticklabels=['1 CPU rank','2 CPU ranks'],ylabel=title)
        if key=='test_accuracy':ax.set_ylim(0,1)
    fig.suptitle('Measured Transformer training on real digits, CPU Gloo data parallelism')
    fig.tight_layout()
    fig.savefig(out/'Figure_011_Distributed_Transformer_Parallelism.png',dpi=200)
    plt.close(fig)
    result={
        'dataset_name':'scikit-learn real handwritten digits',
        'dataset_source_url':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html',
        'split_counts':{'train':1257,'validation':270,'test':270},
        'split_disjoint':True,'training_seed':seed,'epochs':epochs,
        'model':'two-head, one-layer Transformer encoder on 8 real digit-row tokens',
        'parallel_strategy':'CPU Gloo DistributedDataParallel, gradients synchronized',
        'world_size_comparisons':observations,
        'limitations':'Only data parallel CPU, not tensor/pipeline/expert/GPU/multi-node. Independent trainings may differ; the sampler pads the two-rank partition. Measured timing includes synchronization and validation.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=4)
    a=p.parse_args()
    run(a.out,a.epochs)
