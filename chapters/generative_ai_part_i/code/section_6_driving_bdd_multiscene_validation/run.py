"""BDD100K conditional generative object-crop augmentation benchmark.

Requires genuine BDD100K 2018 detection annotation JSON and matching photos.
Trains a class-conditional VAE on real object crops; tests real-only versus
real-resampling versus synthetic augmentation on held-out genuine frames.
This is crop classification, NOT object-detection mAP or whole-scene synthesis.
"""
import argparse,csv,hashlib,json,random
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score,f1_score,accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES=("car","person","traffic light")
SHAPE=(64,64)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_crops(jsonpath,root,classes,limit,seed):
    frames=json.loads(Path(jsonpath).read_text())
    if not isinstance(frames,list):raise ValueError("Expected official BDD frame list")
    items=[]
    for frame in frames:
        name=frame.get("name")
        if not isinstance(name,str):continue
        for label in frame.get("labels") or []:
            cat=label.get("category")
            if cat not in classes:continue
            box=label.get("box2d")
            if not box:continue
            try:x0,y0,x1,y1=(float(box[k]) for k in ("x1","y1","x2","y2"))
            except (ValueError,KeyError,TypeError):continue
            if not np.all(np.isfinite((x0,y0,x1,y1))) or min(x1-x0,y1-y0)<32:continue
            items.append((name,cat,(x0,y0,x1,y1)))
    rng=np.random.default_rng(seed);selected=[]
    for cat in classes:
        pool=[q for q in items if q[1]==cat]
        selected.extend(pool[j] for j in rng.permutation(len(pool))[:limit])
    result=[];missing=[]
    for name,cat,box in selected:
        filename=Path(root)/name
        if not filename.is_file():missing.append(str(filename));continue
        with Image.open(filename) as im:
            rgb=im.convert("RGB")
            x0,y0,x1,y1=box
            bounds=(max(0,int(x0)),max(0,int(y0)),min(rgb.width,int(x1)),min(rgb.height,int(y1)))
            if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:continue
            crop=ImageOps.fit(rgb.crop(bounds),SHAPE,method=Image.Resampling.BICUBIC)
            result.append((name,cat,np.asarray(crop,dtype=np.uint8)))
    if missing:raise FileNotFoundError(f"{len(missing)} real labeled source frames missing; example {missing[0]}")
    return result

def split_data(training,test):
    names=sorted({x[0] for x in training})
    fitnames,valnames=train_test_split(names,test_size=.15,random_state=20261110)
    fitset,valset=set(fitnames),set(valnames)
    fit=[x for x in training if x[0] in fitset]
    val=[x for x in training if x[0] in valset]
    if fitset&valset or ({x[0] for x in test}&(fitset|valset)):
        raise ValueError("Parent-frame leakage across splits")
    groups=[]
    for sample in (fit,val,test):
        if not sample:raise ValueError("Empty genuine data split")
        counts=Counter(x[1] for x in sample)
        if min(counts.get(k,0) for k in CLASSES)<20:
            raise ValueError(f"Insufficient real crops per class: {counts}")
        groups.append({hashlib.sha256(x[2].tobytes()).hexdigest() for x in sample})
    if any(groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError("Identical crop pixels leak across splits")
    return fit,val,test

def tensorize(data):
    x=np.stack([v[2].transpose(2,0,1) for v in data]).astype("float32")/255.
    y=np.array([CLASSES.index(v[1]) for v in data],dtype=np.int64)
    return torch.from_numpy(x),torch.from_numpy(y)

class ClassCVAE(nn.Module):
    def __init__(self,z=32):
        super().__init__();self.z=z
        self.encoder=nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.ReLU(),
          nn.Conv2d(32,48,4,2,1),nn.ReLU(),nn.Conv2d(48,64,4,2,1),nn.ReLU(),nn.Flatten())
        self.mu=nn.Linear(64*8*8+3,z);self.lv=nn.Linear(64*8*8+3,z)
        self.projection=nn.Linear(z+3,64*8*8)
        self.decoder=nn.Sequential(nn.Unflatten(1,(64,8,8)),
          nn.ConvTranspose2d(64,48,4,2,1),nn.ReLU(),
          nn.ConvTranspose2d(48,32,4,2,1),nn.ReLU(),
          nn.ConvTranspose2d(32,3,4,2,1),nn.Sigmoid())
    def generate(self,z,y):
        return self.decoder(self.projection(torch.cat((z,nn.functional.one_hot(y,3).float()),1)))
    def forward(self,x,y):
        h=torch.cat((self.encoder(x),nn.functional.one_hot(y,3).float()),1)
        mu=self.mu(h);lv=self.lv(h).clamp(-8,5)
        return self.generate(mu+torch.randn_like(mu)*torch.exp(.5*lv),y),mu,lv

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,24,3,2,1),nn.ReLU(),
          nn.Conv2d(24,48,3,2,1),nn.ReLU(),nn.Conv2d(48,64,3,2,1),nn.ReLU(),
          nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,3))
    def forward(self,x):return self.net(x)

def train_classifier(x,y,seed,epochs,batch):
    torch.manual_seed(seed)
    model=Classifier().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.002)
    for epoch in range(epochs):
        model.train()
        for ids in torch.randperm(len(x)).split(batch):
            result=model(x[ids].to(DEVICE))
            loss=nn.functional.cross_entropy(result,y[ids].to(DEVICE))
            opt.zero_grad();loss.backward();opt.step()
    return model.eval()

def eval_classifier(model,x,y):
    model.eval()
    with torch.no_grad():
        probs=torch.cat([model(t.to(DEVICE)).softmax(-1).cpu() for t in x.split(128)]).numpy()
    truth=y.numpy();pred=probs.argmax(1)
    return dict(accuracy=float(accuracy_score(truth,pred)),
      balanced_accuracy=float(balanced_accuracy_score(truth,pred)),
      macro_f1=float(f1_score(truth,pred,average="macro"))),probs

def main():
    p=argparse.ArgumentParser()
    for name in ("train-images","val-images","train-labels","val-labels"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--output",type=Path,default=Path("bdd_real_result"))
    p.add_argument("--per-class",type=int,default=500)
    p.add_argument("--vae-epochs",type=int,default=10)
    p.add_argument("--classifier-epochs",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=48)
    p.add_argument("--audit-only",action="store_true")
    args=p.parse_args()
    if args.per_class<50:raise SystemExit("HOLD: minimum per-class crop cap 50")
    for filename in (args.train_images,args.val_images,args.train_labels,args.val_labels):
        if not filename.exists():raise SystemExit(f"HOLD: genuine BDD100K input missing: {filename}")
    raw_train=read_crops(args.train_labels,args.train_images,CLASSES,args.per_class,20261110)
    raw_test=read_crops(args.val_labels,args.val_images,CLASSES,args.per_class,20261111)
    train,val,test=split_data(raw_train,raw_test)
    manifest={"dataset":"BDD100K 2018 object-detection photo/label pairs",
      "train_label_sha256":hashlib.sha256(args.train_labels.read_bytes()).hexdigest(),
      "val_label_sha256":hashlib.sha256(args.val_labels.read_bytes()).hexdigest(),
      "classes":CLASSES,"n_train":len(train),"n_validation":len(val),"n_test":len(test),
      "n_train_parent_frames":len({q[0] for q in train}),
      "n_val_parent_frames":len({q[0] for q in val}),
      "n_test_parent_frames":len({q[0] for q in test}),
      "limitation":"road-object crop classification NOT scene synthesis or detection mAP",
      "sampling":"real-only versus bootstrap real versus CVAE-generated"}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    if args.audit_only:
        print(json.dumps({"status":"INPUT_AUDIT_ONLY",**manifest},indent=2));return
    tx,ty=tensorize(train);vx,vy=tensorize(val);testx,testy=tensorize(test)
    scores=[];predictions=[]
    for seed in (11,23,37):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        generator=ClassCVAE().to(DEVICE);opt=torch.optim.Adam(generator.parameters(),lr=.001)
        best=float("inf");state=None
        for epoch in range(args.vae_epochs):
            generator.train()
            for ids in torch.randperm(len(tx)).split(args.batch_size):
                x=tx[ids].to(DEVICE);y=ty[ids].to(DEVICE)
                reconstructed,mu,lv=generator(x,y)
                kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(1)).mean()
                loss=(reconstructed-x).square().mean()+.00015*kl
                opt.zero_grad();loss.backward();opt.step()
            generator.eval()
            # Deterministic posterior-mean validation (not noise-sampling variance)
            with torch.no_grad():
                values=[]
                for x,y in zip(vx.split(args.batch_size),vy.split(args.batch_size)):
                    x=x.to(DEVICE);y=y.to(DEVICE)
                    h=torch.cat((generator.encoder(x),nn.functional.one_hot(y,3).float()),1)
                    prediction=generator.generate(generator.mu(h),y)
                    values.append((float((prediction-x).square().mean()),len(x)))
                validation=sum(v*n for v,n in values)/sum(n for _,n in values)
            if validation<best:
                best=validation;state={k:v.detach().cpu().clone() for k,v in generator.state_dict().items()}
        if state is None:raise RuntimeError("No trained CVAE checkpoint")
        generator.load_state_dict(state);generator.eval()
        sample_y=ty.clone();synthetic=[]
        with torch.no_grad():
            for ids in torch.arange(len(sample_y)).split(args.batch_size):
                z=torch.randn((len(ids),generator.z),device=DEVICE)
                synthetic.append(generator.generate(z,sample_y[ids].to(DEVICE)).cpu())
        generated=torch.cat(synthetic)
        torch.save({"weights":state,"classes":CLASSES,"seed":seed,
          "train_label_sha256":manifest["train_label_sha256"]},args.output/f"cvae_seed{seed}.pt")
        np_rng=np.random.default_rng(seed)
        for arm in ("real_only","real_bootstrap","real_plus_cvae"):
            if arm=="real_only":x,y=tx,ty
            elif arm=="real_bootstrap":
                ids=torch.tensor(np_rng.integers(0,len(tx),size=len(tx)))
                x=torch.cat((tx,tx[ids]));y=torch.cat((ty,ty[ids]))
            else:x=torch.cat((tx,generated));y=torch.cat((ty,sample_y))
            classifier=train_classifier(x,y,seed,args.classifier_epochs,args.batch_size)
            met,prob=eval_classifier(classifier,testx,testy)
            scores.append(dict(seed=seed,arm=arm,**met,n_training=len(x),n_test=len(test)))
            for i,pv in enumerate(prob):
                predictions.append([seed,arm,i,test[i][0],CLASSES[int(testy[i])],*pv.tolist()])
        if seed==11:
            figure,axes=plt.subplots(3,8,figsize=(12.5,4.6),layout="constrained")
            for classindex in range(3):
                ids=torch.where(sample_y==classindex)[0][:8]
                for j,index in enumerate(ids):
                    axes[classindex,j].imshow(generated[index].permute(1,2,0).numpy().clip(0,1))
                    axes[classindex,j].axis("off")
                    if j==0:axes[classindex,j].set_title("Generated "+CLASSES[classindex],fontsize=8)
            figure.savefig(args.output/"genuine_data_trained_generated_crops.png",dpi=200)
            plt.close(figure)
    with (args.output/"scores.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
    with (args.output/"genuine_test_predictions.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["seed","arm","test_index","real_parent_frame","true_class",
          *["p_"+v.replace(" ","_") for v in CLASSES]])
        w.writerows(predictions)
    figure,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
    for i,arm in enumerate(("real_only","real_bootstrap","real_plus_cvae")):
        vals=[q["macro_f1"] for q in scores if q["arm"]==arm]
        ax.scatter(i+np.array([-.09,0,.09]),vals,s=50)
        ax.hlines(np.mean(vals),i-.2,i+.2,color="#17677D",lw=2)
    ax.set(xticks=range(3),xticklabels=["Real","Real bootstrap","Real + CVAE"],
      ylabel="Macro F1 on held-out real road-object crops",ylim=(0,1.02))
    ax.grid(axis="y",alpha=.2)
    figure.savefig(args.output/"genuine_downstream_macro_f1.png",dpi=200);plt.close(figure)
    print(json.dumps({"status":"EXECUTED_REAL_BDD100K","scores":scores},indent=2))

if __name__=="__main__":main()
