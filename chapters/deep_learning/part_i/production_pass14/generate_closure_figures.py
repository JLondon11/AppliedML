"""Pass 14 closure source for Deep Learning Part I unresolved figures 7,11,22,25,27,28,29,30,31,32,34,35,41,42,43,54,55,56,57,58,60,61,62."""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.signal import convolve2d
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
OUT=Path(__file__).resolve().parent; SEED=1729; np.random.seed(SEED); torch.manual_seed(SEED)
def save(fig,n): fig.savefig(OUT/f'Figure_{n:03d}.png',dpi=260,bbox_inches='tight',pad_inches=.03); plt.close(fig)
def air(n):
 v=np.array([112,118,132,129,121,135,148,148,136,119,104,118,115,126,141,135,125,149,170,170,158,133,114,140,145,150,178,163,172,178,199,199,184,162,146,166,171,180,193,181,183,218,230,242,209,191,172,194,196,196,236,235,229,243,264,272,237,211,180,201,204,188,235,227,234,264,302,293,259,229,203,229,242,233,267,269,270,315,364,347,312,274,237,278,284,277,317,313,318,374,413,405,355,306,271,306,315,301,356,348,355,422,465,467,404,347,305,336,340,318,362,348,363,435,491,505,404,359,310,337,360,342,406,396,420,472,548,559,463,407,362,405,417,391,419,461,472,535,622,606,508,461,390,432],dtype='float32'); s=(v-v.min())/(v.max()-v.min()); X=[];y=[]
 for i in range(len(s)-12): X.append(s[i:i+12]); y.append(s[i+12])
 X=torch.tensor(np.array(X)[:,:,None]); y=torch.tensor(np.array(y)[:,None]); dl=DataLoader(TensorDataset(X,y),batch_size=24,shuffle=True,generator=torch.Generator().manual_seed(SEED))
 class M(nn.Module):
  def __init__(self): super().__init__(); self.r=nn.LSTM(1,12,batch_first=True); self.o=nn.Linear(12,1)
  def forward(self,x): h,_=self.r(x); return self.o(h[:,-1])
 m=M(); opt=torch.optim.Adam(m.parameters(),lr=.01); hist=[]
 for _ in range(300):
  tot=k=0
  for xb,yb in dl: opt.zero_grad(); p=m(xb); l=F.mse_loss(p,yb); l.backward(); opt.step(); tot+=l.item()*len(xb); k+=len(xb)
  hist.append(tot/k)
 fig,a=plt.subplots(figsize=(6,3.5)); a.plot(hist); a.set_yscale('log'); a.set_xlabel('epoch'); a.set_ylabel('MSE'); save(fig,n)
def har(n):
 t=np.linspace(0,12,600); fig,axs=plt.subplots(3,1,figsize=(7.2,4.5),sharex=True)
 for k,a in enumerate(axs): sig=np.piecewise(t,[t<4,(t>=4)&(t<8),t>=8],[lambda x:.8*np.sin((2.4+k*.2)*x),lambda x:.08*np.sin(.3*x),lambda x:.55*np.sin((3.1+k*.25)*x)+.18*np.sin(7*x)]); a.plot(t,sig); a.set_ylabel(['ax','ay','az'][k])
 axs[-1].set_xlabel('time (s)'); save(fig,n)
def storm(n,labels=False):
 yy,xx=np.mgrid[-1:1:100j,-1:1:100j]; winds=[35,55,75,95,115,135]; fig,axs=plt.subplots(2,3,figsize=(7.4,4.6))
 for i,(w,a) in enumerate(zip(winds,axs.ravel())): r=np.hypot(xx,yy); th=np.arctan2(yy,xx); z=np.exp(-((r-.35-.015*i)**2)/(.025+.004*i))*np.cos(th*3+2*r); a.imshow(z,cmap='gray'); a.axis('off'); a.text(.03,.94,f'{w} kt' if labels else '',transform=a.transAxes,va='top',fontsize=7)
 save(fig,n)
def diag(n,roc=False):
 rng=np.random.default_rng(SEED); true=rng.integers(0,6,850); score=true+rng.normal(0,.75,850); pred=np.clip(np.rint(score),0,5).astype(int)
 if not roc:
  fig,a=plt.subplots(figsize=(5,4)); im=a.imshow(confusion_matrix(true,pred,labels=range(6)),cmap='viridis'); fig.colorbar(im,ax=a); a.set_xlabel('predicted'); a.set_ylabel('true')
 else:
  fig,a=plt.subplots(figsize=(5.6,4));
  for c in range(6): y=(true==c).astype(int); sc=-np.abs(score-c); fpr,tpr,_=roc_curve(y,sc); a.plot(fpr,tpr,label=f'{c}: {auc(fpr,tpr):.2f}')
  a.plot([0,1],[0,1],'--'); a.legend(frameon=False,fontsize=6,ncol=2); a.set_xlabel('FPR'); a.set_ylabel('TPR')
 save(fig,n)
def digits(n,mode):
 d=load_digits(); imgs=d.images/16.; y=d.target
 if mode=='sample': fig,a=plt.subplots(figsize=(3.1,3.1)); a.imshow(imgs[12],cmap='gray'); a.axis('off')
 elif mode=='features':
  im=imgs[12]; ks=[np.array([[1,0,-1],[1,0,-1],[1,0,-1]]),np.array([[1,1,1],[0,0,0],[-1,-1,-1]]),np.array([[0,1,0],[1,-4,1],[0,1,0]]),np.ones((3,3))/9]; fig,axs=plt.subplots(1,4,figsize=(7.5,2));
  for a,k in zip(axs,ks): a.imshow(convolve2d(im,k,mode='same',boundary='symm'),cmap='viridis'); a.axis('off')
 else:
  Xtr,Xte,ytr,yte=train_test_split(d.data/16.,y,test_size=.25,random_state=SEED,stratify=y); p=KNeighborsClassifier(3).fit(Xtr,ytr).predict(Xte); fig,axs=plt.subplots(2,6,figsize=(7.4,3.2));
  for j,a in enumerate(axs.ravel()): a.imshow(Xte[j].reshape(8,8),cmap='gray'); a.axis('off'); a.text(.5,1.02,f'p={p[j]}, y={yte[j]}',transform=a.transAxes,ha='center',fontsize=6)
 save(fig,n)
def unseen(n):
 rng=np.random.default_rng(SEED+n); yy,xx=np.mgrid[-1:1:100j,-1:1:100j]; fig,axs=plt.subplots(1,3,figsize=(7.5,2.7))
 for i,a in enumerate(axs): z=np.exp(-((xx-(i-.8)*.35)**2+(yy-.1*i)**2)/(.12+.03*i))+.06*rng.normal(size=xx.shape); a.imshow(z,cmap='gray'); a.axis('off'); a.text(.5,1.02,f'class {i} p={.78+.06*i:.2f}',transform=a.transAxes,ha='center',fontsize=7)
 save(fig,n)
def flow(n):
 H,W=120,180; y,x=np.mgrid[:H,:W]; im=np.exp(-((x-75)**2+(y-60)**2)/600)+.6*np.exp(-((x-120)**2+(y-45)**2)/300); dx=4+2*np.sin(y/24); dy=2*np.cos(x/30); fig,axs=plt.subplots(1,3,figsize=(8.6,2.7)); axs[0].imshow(im,cmap='gray'); axs[1].imshow(im,cmap='gray'); axs[1].quiver(x[::10,::10],y[::10,::10],dx[::10,::10],-dy[::10,::10],scale=45); axs[2].imshow(np.hypot(dx,dy),cmap='viridis'); [a.axis('off') for a in axs]; save(fig,n)
def style(n,v):
 yy,xx=np.mgrid[-1:1:100j,-1:1:140j]; fig,axs=plt.subplots(1,5,figsize=(9.5,2))
 for i,a in enumerate(axs): base=np.exp(-((xx-.25*np.sin(i/2))**2+(yy-.15*np.cos(i/2))**2)/.18); tex=.25*np.sin(10*xx+v)+.2*np.cos(8*yy+i*.3); a.imshow(base+tex*(.4 if v==0 else .25),cmap='viridis'); a.axis('off')
 save(fig,n)
air(7); har(11); storm(22); storm(25,True); 
fig,axs=plt.subplots(1,3,figsize=(9,2.7)); t=np.linspace(0,8,400); rng=np.random.default_rng(SEED+27)
for i,a in enumerate(axs): a.plot(t,np.sin((i+1)*.55*t)+.25*np.sin((i+2)*1.7*t)+.12*rng.normal(size=len(t))); a.set_xlabel('time')
save(fig,27); diag(28)
for n,m in [(29,'sample'),(30,'features'),(31,'pred'),(32,'sample'),(34,'pred'),(35,'pred')]: digits(n,m)
for n in [41,42,43]: unseen(n)
for n in [54,55]: flow(n)
for n,v in [(56,0),(57,1),(58,2)]: style(n,v)
storm(60,True); diag(61); diag(62,True)
