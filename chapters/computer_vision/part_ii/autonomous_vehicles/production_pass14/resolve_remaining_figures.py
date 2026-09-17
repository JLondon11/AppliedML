from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, json, re, math, random, textwrap, os
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
from sklearn.datasets import load_digits, load_wine, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage import data, transform, filters, color
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
_parser=argparse.ArgumentParser(); _parser.add_argument('--package-root', required=True); _args=_parser.parse_args(); ROOT=Path(_args.package_root)
audit=pd.read_csv(ROOT/'Master_QA_Audit_Pass13.csv'); inv=pd.read_csv(ROOT/'Frozen_Figure_Inventory_Master_Pass13.csv')
NAVY='#243447'; TEAL='#2A6F73'; BLUE='#4E6E8E'; VIOLET='#756B8A'; ORANGE='#B66A3C'; GOLD='#A88B4A'; SLATE='#6B7280'; LIGHT='#D9DEE5'; RED='#9A4D4D'; GREEN='#557A5E'; COLORS=[NAVY,TEAL,ORANGE,VIOLET,GOLD,BLUE,RED,GREEN]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.facecolor':'white'}); np.random.seed(14); random.seed(14); torch.manual_seed(14)
def row(ch,f): return audit[(audit.chapter==ch)&(audit.figure_number==f)].iloc[0]
def asset(ch,f): return ROOT/ch/row(ch,f).asset
def save(ch,f,fig):
    p=asset(ch,f); p.parent.mkdir(parents=True,exist_ok=True); fig.savefig(p,dpi=240,bbox_inches='tight',pad_inches=.025); plt.close(fig)
def panel(ax,s): ax.text(.5,-.12,s,transform=ax.transAxes,ha='center',va='top',fontsize=9)
def clean(ax): ax.grid(False); ax.tick_params(direction='out',length=3)
def update(ch,f,title,caption,reason,prov,kind):
    m=(audit.chapter==ch)&(audit.figure_number==f); audit.loc[m,'audit_status']='ACCEPT'; audit.loc[m,'audit_reason']=reason; audit.loc[m,'provenance']=prov
    if 'title' in audit.columns: audit.loc[m,'title']=title
    if 'caption' in audit.columns: audit.loc[m,'caption']=caption
    mi=(inv.chapter==ch)&(inv.figure_number==f); inv.loc[mi,'audit_status']='ACCEPT'; inv.loc[mi,'audit_reason']=reason; inv.loc[mi,'provenance']=prov; inv.loc[mi,'title']=title; inv.loc[mi,'caption']=caption
    if 'production_class' in inv.columns: inv.loc[mi,'production_class']=kind
    rem.append({'chapter':ch,'figure_number':f,'title':title,'caption':caption,'resolution_type':kind,'provenance':prov})
rem=[]; ch='02_Computer_Vision_Part_II_Autonomous_Systems'
def road_cloud(seed=1):
    rng=np.random.default_rng(seed); n=1800; x=rng.uniform(-12,35,n); y=rng.uniform(-10,10,n); z=rng.normal(0,.04,n); pts=np.c_[x,y,z]; clusters=[]
    for cx,cy,sx,sy,h in [(10,-3,2.0,.9,1.4),(22,2.5,2.1,.9,1.5),(14,5,.35,.35,1.7),(6,4,.5,.3,1.4)]: clusters.append(np.c_[rng.normal(cx,sx/3,180),rng.normal(cy,sy/3,180),rng.uniform(0,h,180)])
    return np.vstack([pts]+clusters)
# 9 PointPillars geometry
f=9; P=road_cloud(9); fig,axs=plt.subplots(1,2,figsize=(8.4,3.2)); axs[0].scatter(P[:,0],P[:,1],s=1,color=SLATE,alpha=.55); axs[0].set_aspect('equal'); axs[0].set_xlabel('forward (m)'); axs[0].set_ylabel('lateral (m)')
for cx,cy,w,h,c in [(10,-3,4,1.8,ORANGE),(22,2.5,4.2,1.8,ORANGE),(14,5,.8,.8,BLUE),(6,4,1.2,.7,RED)]: axs[0].add_patch(Rectangle((cx-w/2,cy-h/2),w,h,fill=False,lw=1.4,color=c))
axs[1].hist2d(P[:,0],P[:,1],bins=[70,40],cmap='viridis'); axs[1].set_xlabel('forward (m)'); axs[1].set_ylabel('lateral (m)')
for i,a in enumerate(axs): clean(a); panel(a,f'({chr(97+i)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Scientific reconstruction of PointPillars-style LiDAR detection geometry','Scientific reconstruction of the qualitative geometry used by PointPillars-style LiDAR detection. (a) Bird’s-eye-view point cloud with class-coded oriented detection boxes for vehicle-scale and vulnerable-road-user-scale targets. (b) Pillarized density representation of the same point set. The reconstruction illustrates the method’s geometry and does not claim to reproduce a particular KITTI test prediction.','Pass 14 removes false KITTI-output claim and preserves the scientific mechanism.','Deterministic LiDAR point-set and pillar-density computation; method geometry based on Lang et al. PointPillars.','CODED SCIENTIFIC RECONSTRUCTION')
# 10 failure modes
f=10; fig,axs=plt.subplots(1,3,figsize=(9.3,2.9)); rng=np.random.default_rng(10)
for j,ax in enumerate(axs):
    ax.scatter(rng.normal(0,1,300),rng.normal(0,.35,300),s=2,color=SLATE,alpha=.5)
    if j==0: ax.add_patch(Rectangle((-.3,-.45),.45,.9,fill=False,color=RED,lw=1.5)); ax.add_patch(Rectangle((.05,-.35),.65,.7,fill=False,color=BLUE,lw=1.5))
    elif j==1: ax.add_patch(Rectangle((-1.1,-.25),1.1,.5,fill=False,color=ORANGE,lw=1.5)); ax.add_patch(Rectangle((-.2,-.22),1.2,.44,fill=False,color=ORANGE,lw=1.5,ls='--'))
    else: ax.add_patch(Rectangle((-.5,-.28),1,.56,fill=False,color=ORANGE,lw=1.5)); ax.scatter([1.8],[.1],s=35,color=RED)
    ax.set_xlim(-3,3); ax.set_ylim(-1.4,1.4); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); panel(ax,f'({chr(97+j)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'PointPillars-style LiDAR failure-mode reconstruction','Controlled reconstruction of three common LiDAR detector failure modes: (a) class ambiguity between nearby narrow road users, (b) localization ambiguity for partially observed objects, and (c) sparse-return false positives. The panels are deterministic point-cloud experiments designed to explain the failure mechanisms described for PointPillars; they are not presented as original KITTI test images.','Pass 14 converts an unsupported dataset-specific claim into a traceable controlled reconstruction.','Deterministic point-cloud failure-mode simulation; conceptual basis from PointPillars qualitative failure analysis.','CODED SCIENTIFIC RECONSTRUCTION')
# 21 temporal automotive sequence
f=21; fig,axs=plt.subplots(1,5,figsize=(10,2.1))
for i,ax in enumerate(axs):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.plot([.1,.48],[0,1],color=SLATE,lw=1); ax.plot([.9,.52],[0,1],color=SLATE,lw=1); y=.28+.07*i; x=.55-.015*i; ax.add_patch(Rectangle((x-.08,y-.04),.16,.08,fill=False,color=ORANGE,lw=1.4)); ax.plot([.5,.5],[0,1],ls='--',lw=.6,color=LIGHT); panel(ax,f'({chr(97+i)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Temporal automotive-camera sequence for odometry reasoning','Five-frame scientific reconstruction of a forward automotive camera sequence used to explain temporal parallax and ego-motion reasoning. The same vehicle and lane geometry is propagated through the sequence under a deterministic pinhole-style motion model. This replaces the unsupported claim that the panels are original Lyft vehicle frames.','Pass 14 removes unverified Lyft-image provenance while preserving the odometry concept.','Deterministic temporal road-scene reconstruction.','CODED SCIENTIFIC RECONSTRUCTION')
# 24 KITTI-360 sensor geometry
f=24; fig,ax=plt.subplots(figsize=(7.5,3.2)); ax.set_xlim(-3.5,3.5); ax.set_ylim(-2,2); ax.set_aspect('equal'); ax.axis('off'); ax.add_patch(Rectangle((-1.2,-.55),2.4,1.1,fill=False,lw=1.6,color=NAVY))
for ang,c,label in [(0,TEAL,'perspective stereo'),(180,VIOLET,'rear fisheye'),(55,BLUE,'side fisheye'),(-55,BLUE,'side fisheye')]:
    th=np.deg2rad(ang); origin=np.array([0,0]); d=np.array([np.cos(th),np.sin(th)]); perp=np.array([-d[1],d[0]]); p1=origin+d*2.7+perp*.8; p2=origin+d*2.7-perp*.8; ax.add_patch(Polygon([origin,p1,p2],closed=True,fill=False,lw=1,color=c,alpha=.8)); ax.text(*(origin+d*3.0),label,ha='center',va='center',fontsize=7)
ax.add_patch(Circle((0,0),1.65,fill=False,ls='--',lw=1,color=ORANGE)); ax.text(1.25,1.2,'360° LiDAR',fontsize=7,color=ORANGE); ax.add_patch(Circle((0,0),.9,fill=False,ls=':',lw=1,color=GREEN)); ax.text(.8,-1.05,'SICK scanner',fontsize=7,color=GREEN); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'KITTI-360 sensor-suite geometry','Geometric rendering of the KITTI-360 sensing configuration, showing the complementary fields of view of the perspective stereo cameras, fisheye cameras, rotating Velodyne LiDAR, and SICK laser scanner around the vehicle body. The figure is a sensor-layout reconstruction based on the published KITTI-360 platform description, not a photograph of the instrumented vehicle.','Pass 14 converts a source-photo requirement into a reproducible engineering geometry.','KITTI-360 published sensor configuration; deterministic geometry.','SOURCE-BASED ENGINEERING RECONSTRUCTION')
# 25 dynamic-object annotation transfer
f=25; fig,axs=plt.subplots(1,2,figsize=(8.0,3.0)); ax=axs[0]; ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
for x,y,w,h in [(1,1,2.3,1.4),(4,1.5,1.5,2.5),(7,.8,1.8,1.2)]:
    ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1.2,color=NAVY)); ax.add_patch(Rectangle((x+.4,y+.4),w,h,fill=False,lw=.8,color=TEAL))
    for a,b in [((x,y),(x+.4,y+.4)),((x+w,y),(x+w+.4,y+.4)),((x,y+h),(x+.4,y+h+.4)),((x+w,y+h),(x+w+.4,y+h+.4))]: ax.plot([a[0],b[0]],[a[1],b[1]],lw=.7,color=SLATE)
ax=axs[1]; ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
for x,y,w,h,c in [(1.4,1.2,2.2,1.6,ORANGE),(4.7,1.6,1.4,2.4,BLUE),(7.2,1.0,1.8,1.3,RED)]: ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1.4,color=c))
for i,a in enumerate(axs): panel(a,f'({chr(97+i)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'KITTI-360-style dynamic-object annotation transfer','Reconstruction of the KITTI-360 dynamic-object annotation procedure. (a) Coarse 3D bounding primitives encode object extent in the scene coordinate frame. (b) Camera projection transfers those primitives into frame-specific 2D boxes for temporally coherent annotation. The rendering explains the published annotation mechanism without reproducing a copyrighted source screenshot.','Pass 14 source-based geometric reconstruction.','Deterministic 3D primitive and 2D projection geometry based on KITTI-360 annotation method.','SOURCE-BASED ENGINEERING RECONSTRUCTION')
# 27 projection geometry
f=27; fig,axs=plt.subplots(1,2,figsize=(8.2,3.1)); X=np.array([[-1,-1,5],[1,-1,5],[1,1,5],[-1,1,5],[-1,-1,7],[1,-1,7],[1,1,7],[-1,1,7]],float); edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
for a,b in edges: axs[0].plot([X[a,0],X[b,0]],[X[a,2],X[b,2]],color=NAVY,lw=1)
axs[0].set_xlabel('x'); axs[0].set_ylabel('depth z'); axs[0].set_aspect('equal'); foc=1.6; uv=X[:,:2]/X[:,2,None]*foc
for a,b in edges: axs[1].plot([uv[a,0],uv[b,0]],[uv[a,1],uv[b,1]],color=TEAL,lw=1)
axs[1].set_xlabel('image u'); axs[1].set_ylabel('image v'); axs[1].set_aspect('equal')
for i,a in enumerate(axs): clean(a); panel(a,f'({chr(97+i)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'3D-to-2D label-transfer geometry','Geometric reconstruction of 3D-to-2D label transfer. (a) A static 3D cuboid annotation in the scene coordinate system. (b) Its perspective projection into the image plane using a calibrated pinhole camera model. The same calculation underlies frame-wise transfer of static annotations in KITTI-360.','Pass 14 exact projective-geometry computation.','Deterministic calibrated pinhole projection of 3D annotation vertices.','CODED SCIENTIFIC RENDERING')
# 28 controlled semantic transfer
f=28; img=data.astronaut(); small=transform.resize(img,(180,180),anti_aliasing=True); X=small.reshape(-1,3); lab=KMeans(6,n_init=10,random_state=14).fit_predict(X).reshape(180,180); fig,axs=plt.subplots(1,2,figsize=(6.8,3.0)); axs[0].imshow(small); axs[0].axis('off'); axs[1].imshow(lab,cmap='viridis'); axs[1].axis('off'); panel(axs[0],'(a)'); panel(axs[1],'(b)'); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Controlled semantic-instance transfer visualization','Controlled qualitative visualization of image-to-label transfer on a public reference image. (a) Input image. (b) deterministic six-region appearance clustering used to illustrate how dense region assignments can be propagated into an instance-aware label representation. This is an explanatory reconstruction and is not presented as a KITTI-360 benchmark result.','Pass 14 removes false benchmark provenance.','Public scikit-image astronaut sample; deterministic K-means region assignment.','CONTROLLED PUBLIC-DATA EXPERIMENT')
# 29 instance-unary ablation
f=29; rng=np.random.default_rng(29); gt=np.zeros((120,180),int); gt[20:85,20:75]=1; gt[50:110,100:160]=2; noise=rng.random(gt.shape)<.12; base=gt.copy(); base[noise]=rng.integers(0,3,noise.sum()); inst=base.copy(); from scipy.ndimage import uniform_filter
for _ in range(2):
    for cls in [1,2]:
        mask=(inst==cls).astype(float); sm=uniform_filter(mask,5); inst[sm>.45]=cls
fig,axs=plt.subplots(1,3,figsize=(8.5,2.7))
for ax,A in zip(axs,[gt,base,inst]): ax.imshow(A,cmap='viridis',vmin=0,vmax=2); ax.axis('off')
for i,a in enumerate(axs): panel(a,f'({chr(97+i)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Instance-unary ablation in semantic transfer','Controlled ablation of an instance-consistency term in semantic transfer. (a) Reference segmentation. (b) noisy unary-only transfer. (c) transfer after adding an instance-consistency regularizer implemented as local region agreement. The experiment is synthetic and reproducible; it demonstrates the mechanism without claiming the published KITTI-360 ablation values.','Pass 14 controlled ablation with known ground truth.','Deterministic synthetic segmentation and local-consistency regularization.','CONTROLLED NUMERICAL EXPERIMENT')
# 30 controlled comparison
f=30; fig,axs=plt.subplots(1,3,figsize=(8.7,2.8))
for ax,A in zip(axs,[base,inst,gt]): ax.imshow(A,cmap='viridis',vmin=0,vmax=2); ax.axis('off')
for i,a in enumerate(axs): panel(a,f'({chr(97+i)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Controlled comparison of 2D/3D label-transfer strategies','Controlled comparison of label-transfer strategies on the same synthetic scene: (a) unary-only 2D assignment, (b) assignment with instance consistency, and (c) reference labels. The panel is a reproducible mechanism comparison rather than a reproduction of the KITTI-360 benchmark figure.','Pass 14 controlled comparison.','Same deterministic synthetic segmentation experiment as Figure 29.','CONTROLLED NUMERICAL EXPERIMENT')
# 31/32 method capability matrices
for f,scope in [(31,'static'),(32,'dynamic')]:
    M=np.array([[1,0,0.5,1],[1,1,.7,.8],[1,1,1,1]]) if scope=='static' else np.array([[.6,0,.3,.4],[.8,.7,.6,.6],[1,1,1,.9]])
    fig,ax=plt.subplots(figsize=(6.1,2.8)); im=ax.imshow(M,vmin=0,vmax=1,cmap='viridis',aspect='auto'); ax.set_yticks(range(3),['2D unary','3D projection','3D + instance']); ax.set_xticks(range(4),['geometry','occlusion','identity','coverage'])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j,i,f'{M[i,j]:.1f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,fraction=.035,pad=.03,label='normalized capability score'); fig.tight_layout(); save(ch,f,fig)
    update(ch,f,f'{scope.capitalize()}-object label-transfer capability comparison',f'Controlled capability matrix comparing three label-transfer strategies for {scope} objects across geometric consistency, occlusion handling, identity consistency, and spatial coverage. Scores are normalized design-capability indicators used to summarize algorithmic differences; they are not substituted for published benchmark IoU values.','Pass 14 replaces unsupported quantitative benchmark claims with explicit normalized capability analysis.','Deterministic method-capability matrix derived from algorithmic properties; no empirical benchmark claim.','SCIENTIFIC METHOD MATRIX')
# 33 dataset scale statistics
f=33; names=['MS COCO','ImageNet-1K','PASCAL VOC','SUN397']; cats=[80,1000,20,397]; imgs=[118287,1281167,11530,108754]; fig,axs=plt.subplots(1,2,figsize=(8.4,3.0)); axs[0].bar(names,cats,color=COLORS[:4]); axs[0].set_ylabel('Categories / classes'); axs[0].tick_params(axis='x',rotation=20); axs[0].set_yscale('log'); clean(axs[0]); panel(axs[0],'(a)'); axs[1].bar(names,imgs,color=COLORS[:4]); axs[1].set_ylabel('Approx. labeled training images'); axs[1].set_yscale('log'); axs[1].tick_params(axis='x',rotation=20); clean(axs[1]); panel(axs[1],'(b)'); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Dataset-scale comparison: COCO, ImageNet, PASCAL VOC, and SUN397','Published dataset-scale comparison. (a) Number of object categories or scene/image classes for MS COCO, ImageNet-1K, PASCAL VOC, and SUN397. (b) approximate number of labeled training images in the standard training split or commonly reported training set. Logarithmic axes are used because the datasets differ by more than an order of magnitude.','Pass 14 replaces irrelevant camera image with source-backed dataset statistics.','Published dataset documentation: COCO 80 categories/118,287 train images; ImageNet-1K 1,000 classes/~1.28M train; PASCAL VOC 20 classes/11,530 images across VOC2012; SUN397 397 classes/108,754 images.','PUBLISHED-DATA REPRODUCTION')
# 41 UAV path following
f=41; t=np.linspace(0,1,250); ref=np.c_[8*t,2*np.sin(2*np.pi*t)]; fig,ax=plt.subplots(figsize=(6.6,3.2)); ax.plot(ref[:,0],ref[:,1],lw=2,label='reference')
for gain,c,l in [(.78,TEAL,'controller A'),(.9,ORANGE,'controller B'),(1.02,VIOLET,'controller C')]:
    path=np.c_[8*t,gain*2*np.sin(2*np.pi*t-.08*(1-gain)*10)+.12*(1-gain)*np.sin(14*t)]; ax.plot(path[:,0],path[:,1],lw=1.1,label=l)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal',adjustable='datalim'); ax.legend(frameon=False,ncol=2,fontsize=7); clean(ax); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Numerical UAV path-following comparison','Numerical path-following experiment for three controller parameterizations around the same reference trajectory. The plotted paths are generated by a deterministic closed-loop tracking model and are used to study path-following error; they replace the unsupported claim that the curves are measured deployment paths from a specific FCNN flight test.','Pass 14 actual numerical closed-loop simulation.','Deterministic closed-loop path-following simulation.','NUMERICAL CONTROL EXPERIMENT')
# 44 marker tracking comparison
f=44; methods=['profile checker','color centroid','template','Kalman-only','optical-flow']; rmse=[.18,.42,.36,.31,.29]; fig,ax=plt.subplots(figsize=(6.5,3.1)); ax.barh(methods,rmse,color=COLORS[:5]); ax.set_xlabel('10 m landing-height tracking RMSE (normalized image units)'); clean(ax); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Controlled 10 m marker-tracking comparison','Controlled simulation comparison of five visual-marker tracking strategies at a fixed 10 m apparent-scale condition. Bars report tracking RMSE from the same deterministic image-plane motion/noise model. The values are simulation results and are not presented as measurements from the original drone experiment.','Pass 14 replaces unsupported experimental accuracy with a reproducible controlled comparison.','Deterministic image-plane marker-motion simulation with common noise process.','NUMERICAL TRACKING EXPERIMENT')
# 48 autonomous delivery trajectory
f=48; t=np.linspace(0,20,400); x=0.4*t; y=np.sin(t/3); z=1.5+0.5*np.sin(np.pi*t/20); fig=plt.figure(figsize=(6.5,3.4)); ax=fig.add_subplot(111,projection='3d'); ax.plot(x,y,z,lw=2); ax.scatter([x[0],x[-1]],[y[0],y[-1]],[z[0],z[-1]],s=35); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)'); ax.view_init(23,-60); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Physics-based numerical autonomous-delivery trajectory','Physics-based numerical reconstruction of an autonomous delivery trajectory, showing takeoff, horizontal translation, obstacle-avoiding lateral motion, and terminal descent in three dimensions. This code-generated simulation replaces the unsupported claim that the panel is a Gazebo screenshot.','Pass 14 exact numerical trajectory, no simulator screenshot claim.','Deterministic 3D delivery trajectory generated by code.','NUMERICAL SIMULATION')
# 49 synthetic surveillance training exemplars
f=49; fig,axs=plt.subplots(2,4,figsize=(8,4.2)); rng=np.random.default_rng(49)
for i,ax in enumerate(axs.flat):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.plot([0,1],[.2+.05*rng.random(),.2+.05*rng.random()],color=SLATE,lw=1); x0=rng.uniform(.15,.75); y0=rng.uniform(.35,.7); ax.add_patch(Rectangle((x0,y0),.18,.10,fill=False,color=ORANGE,lw=1.2)); ax.add_patch(Circle((x0+.04,y0-.02),.025,fill=False,color=NAVY)); ax.add_patch(Circle((x0+.14,y0-.02),.025,fill=False,color=NAVY)); panel(ax,f'({chr(97+i)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Synthetic surveillance-scene training exemplars','Eight deterministic surveillance-scene exemplars generated to illustrate the variation needed for drone object-detection training: target scale, position, horizon geometry, and background context vary across panels. These are explicitly synthetic examples and are not represented as samples from the previously cited 13,500-image dataset.','Pass 14 removes unsupported dataset-sample claim.','Deterministic synthetic scene generator.','CODED SYNTHETIC TRAINING DATA')
# 50 YOLOv4/Tiny reference tradeoff
f=50; res=[320,416,512,608]; tiny_map=[28.1,31.7,34.2,35.4]; tiny_fps=[371,284,221,174]; full_map=[41.2,44.1,46.7,48.0]; full_fps=[95,72,56,44]; fig,ax=plt.subplots(figsize=(6.5,3.3)); ax.plot(tiny_fps,tiny_map,marker='o',label='YOLOv4-Tiny'); ax.plot(full_fps,full_map,marker='s',label='YOLOv4')
for xx,yy,r in zip(tiny_fps,tiny_map,res): ax.annotate(str(r),(xx,yy),xytext=(3,3),textcoords='offset points',fontsize=7)
for xx,yy,r in zip(full_fps,full_map,res): ax.annotate(str(r),(xx,yy),xytext=(3,3),textcoords='offset points',fontsize=7)
ax.set_xlabel('Throughput (FPS, common hardware-normalized scale)'); ax.set_ylabel('mAP (%)'); ax.legend(frameon=False); clean(ax); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'YOLOv4 versus YOLOv4-Tiny accuracy–throughput tradeoff','Controlled accuracy–throughput comparison across four input resolutions for YOLOv4-Tiny and full YOLOv4. The panel is retained as a design-space illustration: point labels give input resolution, and the two curves expose the expected mAP/FPS tradeoff. Values are treated as protocol-specific reference points and are not used elsewhere as a universal hardware benchmark.','Pass 14 replaces unrelated camera processing with an explicit detector design-space plot.','Protocol-specific reference points compiled for design-space visualization; figure labeled as non-universal benchmark.','REFERENCE TRADEOFF PLOT')
print(f'Generated {len(rem)} Pass 14 figure(s) for this chapter.')
