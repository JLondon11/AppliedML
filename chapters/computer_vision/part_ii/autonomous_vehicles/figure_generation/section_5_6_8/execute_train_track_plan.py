"""Deterministic *generated-scene* training, Kalman tracking and lane-planning test.
NOT autonomous-driving dataset inference, field testing or safety validation.
Requires numpy, scipy, scikit-learn, matplotlib.
Run from its directory: python execute_train_track_plan.py.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent
FIG=ROOT/'Figures'; FIG.mkdir(exist_ok=True)
SEED=56831
train_rng=np.random.default_rng(SEED)
test_rng=np.random.default_rng(SEED+1)
LANES=np.array([-2.6,0.,2.6]); HORIZON=3.; EGO_SPEED=6.
def feature_sample(rng,positive):
    if positive:
        return np.array([rng.normal(.70,.20),rng.normal(1.60,.42),rng.normal(.72,.15)])
    return np.array([rng.normal(.39,.25),rng.normal(.78,.49),rng.normal(.40,.20)])
Xtrain=np.array([feature_sample(train_rng,i%2==0) for i in range(12000)])
ytrain=np.array([int(i%2==0) for i in range(12000)])
classifier=make_pipeline(StandardScaler(),LogisticRegression(max_iter=400,random_state=SEED))
classifier.fit(Xtrain,ytrain)
Xtest=np.array([feature_sample(test_rng,i%2==0) for i in range(4000)])
ytest=np.array([int(i%2==0) for i in range(4000)])
proba=classifier.predict_proba(Xtest)[:,1]
labels=proba>=.5
base_labels=(Xtest[:,0]>=.55)&(Xtest[:,1]>=1.1)
def eval_binary(yhat):
    p,r,f,_=precision_recall_fscore_support(ytest,yhat,average='binary',zero_division=0)
    return p,r,f
p0,r0,f0=eval_binary(base_labels)
p1,r1,f1=eval_binary(labels)
auc=roc_auc_score(ytest,proba)
DT=.25
F=np.array([[1,0,DT,0],[0,1,0,DT],[0,0,1,0],[0,0,0,1]],float)
H=np.array([[1,0,0,0],[0,1,0,0]],float)
Q=np.diag([.015,.015,.07,.07]); R=np.diag([.27**2,.40**2]); I=np.eye(4)
class Track:
    def __init__(self,z):
        self.s=np.array([z[0],z[1],0.,0.])
        self.P=np.diag([.2,.2,15.,15.]); self.age=0
    def predict(self):
        self.s=F@self.s
        self.P=F@self.P@F.T+Q; self.age+=1
    def update(self,z):
        nu=z-H@self.s;S=H@self.P@H.T+R
        K=self.P@H.T@np.linalg.inv(S);self.s=self.s+K@nu
        A=I-K@H;self.P=A@self.P@A.T+K@R@K.T;self.age=0
def tracks_from_frames(frames):
    tr=[]
    for frame in frames:
        for t in tr:t.predict()
        if not frame:continue
        zs=np.asarray(frame);valid=np.zeros(len(zs),bool)
        if tr:
            dist=np.linalg.norm(np.stack([H@t.s for t in tr])[:,None,:]-zs[None,:,:],axis=2)
            ii,jj=linear_sum_assignment(dist);matched=set()
            for i,j in zip(ii,jj):
                if dist[i,j]<3.8:
                    tr[i].update(zs[j]);matched.add(j)
            valid[list(matched)]=True
        for j,z in enumerate(zs):
            if not valid[j]:tr.append(Track(z))
        tr=[t for t in tr if t.age<=2]
    return tr
def generate_scene(rng):
    objs=[]
    for lane in LANES:
        near=(lane==0 or rng.random()<.61)
        y=rng.uniform(13.,23.) if near else rng.uniform(29,43)
        x=lane+rng.normal(0,.22)
        vx=rng.uniform(-.75,.75) if lane!=0 else rng.uniform(-.25,.25)
        vy=rng.uniform(-.8,1.2)
        objs.append(np.array([x,y,vx,vy]))
    return objs
def make_frames(rng,objs):
    frames=[];last_observed=[]
    for t in np.arange(-1.25,.001,DT):
        detects=[]
        for o in objs:
            if rng.random()<.09:continue
            feat=feature_sample(rng,True)
            score=classifier.predict_proba(feat.reshape(1,-1))[0,1]
            if score>=.5:
                observed=o[:2]+t*o[2:]+rng.normal([0,0],[.27,.40])
                detects.append(observed)
        for _ in range(rng.poisson(2)):
            feat=feature_sample(rng,False)
            score=classifier.predict_proba(feat.reshape(1,-1))[0,1]
            if score>=.5:
                detects.append([rng.uniform(-5.2,5.2),rng.uniform(8,39)])
        frames.append(detects)
        if abs(t)<1e-10:last_observed=detects
    return frames,last_observed
def lane_choice(estimates,predict):
    times=np.arange(.5,HORIZON+.001,.5);costs=[]
    for lane in LANES:
        c=.15*abs(lane)+(.3 if lane!=0 else 0)
        for t in times:
            for s in estimates:
                ox=s[0]+(s[2]*t if predict else 0)
                oy=s[1]+(s[3]*t if predict else 0)
                lateral=abs(ox-lane);longitudinal=abs(oy-EGO_SPEED*t)
                c+=23*np.exp(-.5*((lateral/1.)**2+(longitudinal/3.5)**2))
        costs.append(c)
    return int(np.argmin(costs))
def collision(lane,objs):
    times=np.linspace(0,HORIZON,121);worst=1.e6
    for o in objs:
        for t in times:
            dx=(o[0]+o[2]*t)-LANES[lane]
            dy=(o[1]+o[3]*t)-EGO_SPEED*t
            gap=(dx/1.25)**2+(dy/2.7)**2
            worst=min(worst,gap)
    return worst<1.,worst
N=160
all_results=[];tracking_error=[];illustration=None
for episode in range(N):
    objs=generate_scene(test_rng)
    frames,raw=make_frames(test_rng,objs)
    tracks=tracks_from_frames(frames)
    states=[t.s for t in tracks if t.age==0]
    raw_states=[np.array([z[0],z[1],0.,0.]) for z in raw]
    static=lane_choice(raw_states,False);predicted=lane_choice(states,True)
    coll_static,sep_static=collision(static,objs)
    coll_pred,sep_pred=collision(predicted,objs)
    all_results.append([episode,len(states),len(raw),static,predicted,
                        int(coll_static),int(coll_pred),sep_static,sep_pred])
    if states:
        for ob in objs:
            nearest=min(states,key=lambda s:np.linalg.norm(s[:2]-ob[:2]))
            if np.linalg.norm(nearest[:2]-ob[:2])<3.:
                tracking_error.append((nearest[:2]-ob[:2])**2)
    if illustration is None and static!=predicted and not coll_pred:
        illustration=(objs,raw_states,states,static,predicted)
arr=np.asarray(all_results)
static_rate=arr[:,5].mean();track_rate=arr[:,6].mean()
rmse=np.sqrt(np.mean(tracking_error)) if tracking_error else np.nan
metrics=[
 ('heldout_detection_f1_fixed_threshold',f0),
 ('heldout_detection_f1_trained_logistic',f1),
 ('heldout_detection_auc_trained_logistic',auc),
 ('track_position_rmse_m_per_axis_matched',rmse),
 ('collision_rate_reactive',static_rate),
 ('collision_rate_tracked_predictive',track_rate),
 ('evaluation_episodes',N),
 ('train_samples',len(ytrain)),
 ('test_samples',len(ytest))]
with (ROOT/'train_track_plan_metrics.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['metric','value']);w.writerows(metrics)
with (ROOT/'train_track_plan_episodes.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['episode','tracks','raw_detections',
      'reactive_lane','predictive_lane','reactive_collision','predictive_collision',
      'reactive_min_normalized_gap2','predictive_min_normalized_gap2'])
    w.writerows(all_results)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10.2,3.25),facecolor='white')
axes[0].bar([0,1],[f0,f1],color=['#71869a','#137c79'],width=.55)
axes[0].set_xticks([0,1],['Fixed rule','Trained classifier'])
axes[0].set_ylabel('Held-out detection F1');axes[0].set_ylim(0,1)
axes[0].text(.02,.96,'(a)',transform=axes[0].transAxes,va='top',fontweight='bold')
for j,v in enumerate([f0,f1]):
    axes[0].text(j,v+.015,f'{v:.3f}',ha='center',fontsize=8)
axes[1].bar([0,1],[static_rate,track_rate],color=['#b96a43','#137c79'],width=.55)
axes[1].set_xticks([0,1],['Reactive','Tracked prediction'])
axes[1].set_ylabel('Simulated collision fraction')
axes[1].set_ylim(0,max(.15,static_rate,track_rate)*1.27)
axes[1].text(.02,.96,'(b)',transform=axes[1].transAxes,va='top',fontweight='bold')
for j,v in enumerate([static_rate,track_rate]):
    axes[1].text(j,v+.013,f'{v:.3f}',ha='center',fontsize=8)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.subplots_adjust(left=.09,right=.97,bottom=.25,top=.93,wspace=.35)
for ext in ('png','pdf','svg'):
    fig.savefig(FIG/f'train_track_plan_results.{ext}',dpi=250,facecolor='white')
plt.close(fig)
fig,ax=plt.subplots(figsize=(7.4,4),facecolor='white')
if illustration is None:
    objs=generate_scene(np.random.default_rng(SEED+2));static=0;predicted=1
else:
    objs,_,_,static,predicted=illustration
for k,lc in enumerate(LANES):
    ax.axvline(lc,color='#b8c3cb',ls=':',lw=1.1)
    ax.text(lc,32,['left','center','right'][k],ha='center',
            fontsize=8,color='#4a5969')
for i,o in enumerate(objs):
    ts=np.linspace(0,HORIZON,50)
    ax.plot(o[0]+o[2]*ts,o[1]+o[3]*ts,color='#c76530',lw=1.5)
    ax.plot(o[0],o[1],'o',color='#c76530',ms=5)
ax.plot([LANES[static]]*50,EGO_SPEED*np.linspace(0,HORIZON,50),
        color='#a24b29',ls='--',lw=2,label='reactive candidate')
ax.plot([LANES[predicted]]*50,EGO_SPEED*np.linspace(0,HORIZON,50),
        color='#087e79',lw=2,label='tracked candidate')
ax.plot(0,0,marker='^',ms=9,color='#172c44')
ax.set(xlim=(-4.4,4.4),ylim=(-1,34),xlabel='Lateral position x (m)',
       ylabel='Longitudinal position y (m)')
ax.legend(loc='upper right',fontsize=8)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ('png','pdf','svg'):
    fig.savefig(FIG/f'train_track_plan_scene.{ext}',dpi=250,facecolor='white')
plt.close(fig)
print({k:round(float(v),5) for k,v in metrics})
