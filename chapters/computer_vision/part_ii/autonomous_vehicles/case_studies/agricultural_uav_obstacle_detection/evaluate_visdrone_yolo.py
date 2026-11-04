"""Real-image VisDrone YOLO evaluation; local AP50, not official challenge AP.

Requires real VisDrone2019-DET-val/ images & annotations, a ten-class
VisDrone-fine-tuned Ultralytics YOLO checkpoint, numpy, cv2, matplotlib.
Neither missing images nor missing checkpoint is ever synthesized.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

CLASSES=('pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()

def iou(box,others):
    other=np.asarray(others,dtype=float).reshape(-1,4)
    if not len(other):return np.empty(0)
    lo=np.maximum(box[:2],other[:,:2])
    hi=np.minimum(box[2:4],other[:,2:4])
    area=np.prod(np.maximum(hi-lo,0),axis=1)
    a=np.prod(np.maximum(np.asarray(box[2:4])-box[:2],0))
    b=np.prod(np.maximum(other[:,2:]-other[:,:2],0),axis=1)
    return area/np.maximum(a+b-area,1e-12)

def parse_visdrone_annotation(path):
    gt,ignored=[],[]
    for line_no,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip():continue
        vals=line.strip().split(',')
        if len(vals)<8:raise ValueError(f'Bad VisDrone annotation {path}:{line_no}')
        x,y,w,h,score,category,trunc,occl=map(float,vals[:8])
        if w<=0 or h<=0:continue
        if not 0<=category<=11:raise ValueError('Unexpected VisDrone class')
        box=[x,y,x+w,y+h]
        if int(category) in (0,11):ignored.append(box)
        else:gt.append(box+[int(category)-1])
    return np.asarray(gt,dtype=float).reshape(-1,5),np.asarray(ignored,dtype=float).reshape(-1,4)

def ap50_for_class(records,c):
    gts={r['id']:r['gt'][r['gt'][:,4]==c,:4] for r in records}
    n_gt=sum(map(len,gts.values()))
    if not n_gt:return float('nan'),0,0,0
    det=sorted([(float(b[4]),r['id'],b[:4]) for r in records for b in r['pred'] if int(b[5])==c],
               key=lambda t:-t[0])
    used={key:set() for key in gts}
    tps=[];fps=[]
    for score,key,box in det:
        overlap=iou(box,gts[key])
        if len(overlap) and float(np.max(overlap))>=.5 and int(np.argmax(overlap)) not in used[key]:
            used[key].add(int(np.argmax(overlap)));tps.append(1);fps.append(0)
        else:tps.append(0);fps.append(1)
    tp=np.cumsum(tps,dtype=float);fp=np.cumsum(fps,dtype=float)
    recall=tp/n_gt; precision=tp/np.maximum(tp+fp,1e-12)
    mr=np.r_[0,recall,1];mp=np.r_[0,precision,0]
    mp=np.maximum.accumulate(mp[::-1])[::-1]
    points=np.where(mr[1:]!=mr[:-1])[0]
    ap=float(np.sum((mr[points+1]-mr[points])*mp[points+1]))
    return ap,n_gt,len(det),int(tp[-1]) if len(tp) else 0

def plot(img,gt,pred,out):
    import cv2,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),facecolor='white')
    for a in axes:
        a.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));a.axis('off')
    for x1,y1,x2,y2,c in gt:
        axes[0].add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#12847e',lw=.9))
    for x1,y1,x2,y2,score,c in pred:
        if score<.25:continue
        axes[1].add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#c56d3e',lw=.9))
        axes[1].text(x1,y1,CLASSES[int(c)]+f' {score:.2f}',fontsize=5,color='#9a4e25')
    for j,a in enumerate(axes):
        a.text(.5,-.05,f'({"ab"[j]})',transform=a.transAxes,ha='center',color='black')
    fig.subplots_adjust(left=.01,right=.99,bottom=.12,top=.99,wspace=.03)
    for ext in ['pdf','png','svg']:
        fig.savefig(out.with_suffix('.'+ext),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    for flag in ('checkpoint','images','annotations','output'):
        ap.add_argument('--'+flag,type=Path,required=True)
    ap.add_argument('--max-images',type=int,default=0)
    ap.add_argument('--plot-first',type=int,default=6)
    ap.add_argument('--confidence',type=float,default=.001)
    a=ap.parse_args()
    if not a.checkpoint.is_file() or a.checkpoint.stat().st_size<100000:
        raise FileNotFoundError('Actual VisDrone-fine-tuned YOLO checkpoint required')
    if not a.images.is_dir() or not a.annotations.is_dir():
        raise FileNotFoundError('Original VisDrone validation images and labels required')
    images=sorted(p for p in a.images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if a.max_images:images=images[:a.max_images]
    if not images:raise FileNotFoundError('No original images')
    for image in images:
        if not (a.annotations/(image.stem+'.txt')).is_file():
            raise FileNotFoundError('Missing original annotation for '+image.name)
    import cv2
    import ultralytics
    from ultralytics import YOLO
    model=YOLO(str(a.checkpoint))
    names=model.names
    normalized=[str(names[k]).strip().lower().replace('_','-') for k in range(len(names))]
    if tuple(normalized)!=CLASSES:
        raise ValueError('Checkpoint must use EXACT VisDrone class ordering; COCO-only weights are invalid')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'figures').mkdir(exist_ok=True)
    records=[]
    with (a.output/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image','class_id','confidence','x1','y1','x2','y2'])
        for i,image_path in enumerate(images):
            gt,ignored=parse_visdrone_annotation(a.annotations/(image_path.stem+'.txt'))
            img=cv2.imread(str(image_path))
            if img is None:raise ValueError('Unreadable original image')
            result=model.predict(source=img,conf=a.confidence,verbose=False)[0]
            pred=np.empty((0,6))
            if result.boxes is not None and len(result.boxes):
                pred=np.column_stack((result.boxes.xyxy.cpu().numpy(),
                                      result.boxes.conf.cpu().numpy(),
                                      result.boxes.cls.cpu().numpy()))
            for b in pred:
                w.writerow([image_path.name,int(b[5]),float(b[4]),*map(float,b[:4])])
            records.append(dict(id=image_path.name,gt=gt,ignored=ignored,pred=pred))
            if i<a.plot_first:plot(img,gt,pred,a.output/'figures'/('visdrone_'+image_path.stem))
    metrics=[(CLASSES[k],*ap50_for_class(records,k)) for k in range(10)]
    with (a.output/'per_class_ap50.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['class','local_AP50','gt_count','prediction_count','true_positives']);w.writerows(metrics)
    scores=[m[1] for m in metrics if np.isfinite(m[1])]
    manifest={'dataset':'Official VisDrone2019-DET-val','n_images':len(images),
      'checkpoint_sha256':sha256(a.checkpoint),'ultralytics_version':ultralytics.__version__,
      'local_mAP50':float(np.mean(scores)) if scores else None,
      'warning':'LOCAL AP50 ONLY. Ignored-region treatment not official VisDrone toolkit; not challenge leaderboard AP.'}
    (a.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
