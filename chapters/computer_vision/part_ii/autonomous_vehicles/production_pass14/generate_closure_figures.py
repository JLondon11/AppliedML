"""Pass 14 closure source for Computer Vision Part II.
Figures: 9,10,21,24,25,27,28,29,30,31,32,33,41,44,48,49,50.
All outputs are controlled simulations or descriptive dataset metadata; no unsupported KITTI/Lyft/Gazebo benchmark claims are made.
"""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT=Path(__file__).resolve().parent; SEED=1729
NAVY='#19324d'; SLATE='#60748a'; TEAL='#2f7f78'; ORANGE='#c87533'; VIOLET='#7a6fa8'; GRAY='#6c737f'
def save(fig,n): fig.savefig(OUT/f'Figure_{n:03d}.png',dpi=260,bbox_inches='tight',pad_inches=.03); plt.close(fig)
def pc(n):
 rng=np.random.default_rng(SEED+n); fig,axs=plt.subplots(1,2,figsize=(7.4,3)); pts=[]
 for cx,cy,sx,sy,k in [(-2,1,.8,.35,180),(1.5,-1,.55,.25,120),(2.2,1.8,.35,.25,90)]: pts.append(rng.normal([cx,cy],[sx,sy],size=(k,2)))
 P=np.vstack(pts+[rng.uniform([-4,-3],[4,3],size=(300,2))]); axs[0].scatter(P[:,0],P[:,1],s=3,alpha=.55,color=SLATE); axs[1].scatter(P[:,0],P[:,1],s=3,alpha=.35,color=SLATE)
 for x,y,w,h,l in [(-3,.25,2,1.5,'vehicle'),(.7,-1.6,1.7,1,'cyclist'),(1.65,1.25,1.1,1,'pedestrian')]: axs[1].add_patch(Rectangle((x,y),w,h,fill=False,lw=1.4,edgecolor=TEAL)); axs[1].text(x,y+h+.08,l,fontsize=7)
 for a in axs: a.set_aspect('equal'); a.set_xlabel('x (m)'); a.set_ylabel('y (m)'); a.spines[['top','right']].set_visible(False)
 save(fig,n)
def seq(n):
 rng=np.random.default_rng(SEED+n); Y,X=np.mgrid[:80,:120]; fig,axs=plt.subplots(1,5,figsize=(10,2.1))
 for i,a in enumerate(axs):
  cx=34+7*i; cy=38+2*i; im=.15+.25*np.exp(-((X-25)**2+(Y-52)**2)/850)+.5*np.exp(-((X-cx)**2+(Y-cy)**2)/160)+.08*np.sin((X+i*2)/7)+.03*rng.normal(size=(80,120)); a.imshow(im,cmap='gray'); a.axis('off')
 save(fig,n)
def project(n):
 rng=np.random.default_rng(SEED+n); xyz=np.c_[rng.uniform(-3,3,160),rng.uniform(-1.5,1.5,160),rng.uniform(6,16,160)]; c=(xyz[:,0]>0).astype(int)+(xyz[:,1]>.3).astype(int); u=320+260*xyz[:,0]/xyz[:,2]; v=180-260*xyz[:,1]/xyz[:,2]
 fig=plt.figure(figsize=(7.5,3)); a=fig.add_subplot(121,projection='3d'); a.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=c,cmap='viridis',s=9); b=fig.add_subplot(122); b.scatter(u,v,c=c,cmap='viridis',s=12); b.invert_yaxis(); b.set_xlabel('image u'); b.set_ylabel('image v'); save(fig,n)
def seg(n):
 rng=np.random.default_rng(SEED+n); H,W=100,150; yy,xx=np.mgrid[:H,:W]; base=.2+.2*np.sin(xx/15)+.15*np.cos(yy/13); m=np.zeros((H,W),int); m[((xx-45)**2/28**2+(yy-55)**2/20**2)<1]=1; m[((xx-105)**2/23**2+(yy-48)**2/26**2)<1]=2; noisy=m.copy(); flip=rng.random((H,W))<(.12 if n%2==0 else .04); noisy[flip]=rng.integers(0,3,flip.sum()); fig,axs=plt.subplots(1,3,figsize=(8.5,2.6)); axs[0].imshow(base,cmap='gray'); axs[1].imshow(noisy,cmap='viridis',vmin=0,vmax=2); axs[2].imshow(m,cmap='viridis',vmin=0,vmax=2); [a.axis('off') for a in axs]; save(fig,n)
def traj(n):
 t=np.linspace(0,18,300); ref=np.c_[t,.8*np.sin(t/2.5)]; p1=ref+np.c_[.12*np.sin(1.7*t),.20*np.sin(.9*t)]; p2=ref+np.c_[.06*np.sin(2*t),.08*np.sin(1.2*t)]; fig,axs=plt.subplots(1,2,figsize=(7.8,2.9)); axs[0].plot(ref[:,0],ref[:,1],label='reference',color=GRAY); axs[0].plot(p1[:,0],p1[:,1],label='A',color=ORANGE); axs[0].plot(p2[:,0],p2[:,1],label='B',color=TEAL); axs[0].legend(frameon=False,fontsize=7); axs[1].plot(t,np.linalg.norm(p1-ref,axis=1),color=ORANGE); axs[1].plot(t,np.linalg.norm(p2-ref,axis=1),color=TEAL); axs[1].set_ylabel('tracking error'); save(fig,n)
for n in [9,10]: pc(n)
seq(21)
for n in [24,25,27]: project(n)
for n in [28,29,30,31,32]: seg(n)
fig,axs=plt.subplots(1,2,figsize=(7.7,3.1)); names=['COCO','ImageNet-1K','PASCAL VOC','SUN397']; axs[0].bar(names,[118000,1281167,11540,108754]); axs[0].set_yscale('log'); axs[1].bar(names,[80,1000,20,397]); axs[1].set_yscale('log'); [a.tick_params(axis='x',rotation=22) for a in axs]; save(fig,33)
for n in [41,44]: traj(n)
fig=plt.figure(figsize=(7,3.8)); a=fig.add_subplot(111,projection='3d');
for x,y,w,d,h in [(-3,-2,1.4,1.2,1.2),(1,-2,1.7,1.1,2),(-1,1,1.2,1.5,1.6),(2,1,1.4,1.2,1)]: a.bar3d(x,y,0,w,d,h,alpha=.45)
t=np.linspace(-4,4,160); a.plot(t,.5*np.sin(t),1.2+.5*np.exp(-t**2/5),lw=2); save(fig,48)
rng=np.random.default_rng(SEED+49); fig,axs=plt.subplots(2,3,figsize=(7.5,4.6)); yy,xx=np.mgrid[:70,:100]
for j,a in enumerate(axs.ravel()):
 im=.2+.12*np.sin(xx/8)+.05*rng.normal(size=(70,100)); cx=20+11*j; cy=38; im+=.55*np.exp(-((xx-cx)**2+(yy-cy)**2)/120); a.imshow(im,cmap='gray'); a.add_patch(Rectangle((cx-10,cy-14),20,28,fill=False,lw=1.2)); a.axis('off')
save(fig,49)
fig,a=plt.subplots(figsize=(5.5,3.7)); fps=[78,52,28,11]; m=[48,57,64,68]; labs=['Tiny','Compact','Full','Two-stage']; a.scatter(fps,m,s=60); [a.text(x+1,y+.4,l,fontsize=7) for x,y,l in zip(fps,m,labs)]; a.set_xlabel('throughput (frames/s)'); a.set_ylabel('mAP-like score (%)'); save(fig,50)
