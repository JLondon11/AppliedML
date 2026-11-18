"""Compare constant, step, cosine, and warmup+cosine learning-rate schedules."""
import argparse, json, math

def main():
    import torch
    from torch import nn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=60); ap.add_argument('--out',default='ce08_results.json'); a=ap.parse_args()
    X,y=load_digits(return_X_y=True); X=X.astype('float32')/16.; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=7)
    Xtr=torch.tensor(Xtr); ytr=torch.tensor(ytr); Xte=torch.tensor(Xte); yte=torch.tensor(yte)
    results=[]
    for name in ['constant','step','cosine','warmup_cosine']:
        torch.manual_seed(7); m=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,10)); opt=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9)
        for e in range(a.epochs):
            if name=='step': lr=.1*(.2**(e//20))
            elif name=='cosine': lr=.1*.5*(1+math.cos(math.pi*e/a.epochs))
            elif name=='warmup_cosine':
                lr=.1*(e+1)/5 if e<5 else .1*.5*(1+math.cos(math.pi*(e-5)/(a.epochs-5)))
            else: lr=.1
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(); loss=nn.functional.cross_entropy(m(Xtr),ytr); loss.backward(); opt.step()
        acc=float((m(Xte).argmax(1)==yte).float().mean()); results.append({'schedule':name,'accuracy':acc})
    open(a.out,'w').write(json.dumps(results,indent=2)); print(results)
if __name__=='__main__': main()
