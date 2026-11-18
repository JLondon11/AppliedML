"""CIFAR-100 comparison with Momentum, AdamW, and the Lion optimizer."""
import argparse, json, time

def main():
    import torch
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    class Lion(Optimizer):
        def __init__(self,params,lr=1e-4,betas=(0.9,0.99),weight_decay=0.0):
            super().__init__(params,dict(lr=lr,betas=betas,weight_decay=weight_decay))
        @torch.no_grad()
        def step(self,closure=None):
            loss=closure() if closure else None
            for group in self.param_groups:
                lr=group['lr']; b1,b2=group['betas']; wd=group['weight_decay']
                for p in group['params']:
                    if p.grad is None: continue
                    g=p.grad
                    if wd: p.mul_(1-lr*wd)
                    st=self.state[p]
                    if 'm' not in st: st['m']=torch.zeros_like(p)
                    m=st['m']; update=m.mul(b1).add(g,alpha=1-b1); p.add_(update.sign(),alpha=-lr); m.mul_(b2).add_(g,alpha=1-b2)
            return loss

    def ece(logits,y,bins=15):
        prob=logits.softmax(1); conf,pred=prob.max(1); acc=pred.eq(y); out=0.0
        edges=torch.linspace(0,1,bins+1,device=logits.device)
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(conf>lo)&(conf<=hi)
            if mask.any(): out+=mask.float().mean()*torch.abs(acc[mask].float().mean()-conf[mask].mean())
        return float(out)

    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch',type=int,default=128); ap.add_argument('--out',default='ce02_results.json'); ap.add_argument('--data',default='./data'); a=ap.parse_args()
    torch.manual_seed(7); device='cuda' if torch.cuda.is_available() else 'cpu'
    tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((.507,.487,.441),(.267,.256,.276))]); te=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.507,.487,.441),(.267,.256,.276))])
    tr=datasets.CIFAR100(a.data,train=True,download=True,transform=tf); va=datasets.CIFAR100(a.data,train=False,download=True,transform=te); tl=DataLoader(tr,batch_size=a.batch,shuffle=True,num_workers=2); vl=DataLoader(va,batch_size=256,num_workers=2)
    class Net(nn.Module):
        def __init__(self): super().__init__(); self.f=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(128,256,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.h=nn.Linear(256,100)
        def forward(self,x): return self.h(self.f(x).flatten(1))
    results=[]
    for name in ['momentum','adamw','lion']:
        torch.manual_seed(7); m=Net().to(device)
        if name=='momentum': opt=torch.optim.SGD(m.parameters(),lr=.05,momentum=.9,weight_decay=5e-4)
        elif name=='adamw': opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=5e-4)
        else: opt=Lion(m.parameters(),lr=3e-4,weight_decay=5e-4)
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        curve=[]; t0=time.perf_counter()
        for ep in range(a.epochs):
            m.train()
            for x,y in tl: x,y=x.to(device),y.to(device); opt.zero_grad(); loss=nn.functional.cross_entropy(m(x),y); loss.backward(); opt.step()
            m.eval(); correct=n=0
            with torch.no_grad():
                for x,y in vl: pred=m(x.to(device)).argmax(1).cpu(); correct+=(pred==y).sum().item(); n+=len(y)
            curve.append(correct/n)
        logits=[]; ys=[]
        with torch.no_grad():
            for x,y in vl: logits.append(m(x.to(device))); ys.append(y.to(device))
        logits=torch.cat(logits); ys=torch.cat(ys); acc=float((logits.argmax(1)==ys).float().mean()); cal=ece(logits,ys); mem=torch.cuda.max_memory_allocated() if device=='cuda' else 0
        results.append({'optimizer':name,'accuracy':acc,'ece':cal,'runtime_s':time.perf_counter()-t0,'peak_cuda_bytes':mem,'accuracy_curve':curve})
    open(a.out,'w').write(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
