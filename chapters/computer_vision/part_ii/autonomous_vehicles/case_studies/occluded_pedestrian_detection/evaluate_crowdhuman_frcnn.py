"""Actual COCO-pretrained torchvision Faster R-CNN on original CrowdHuman ODGT.
Output: per-person matched IoU50 recall stratified by visible/full-body box ratio.
This LOCAL protocol is NOT official CrowdHuman AP/MR/JI evaluation.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import cv2
import numpy as np

def xyxy(v):
    x,y,w,h=map(float,v)
    if w<=0 or h<=0:raise ValueError('Bad annotation box')
    return np.array([x,y,x+w,y+h])

def overlap(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.prod(np.maximum(hi-lo,0),axis=2)
    aa=np.prod(a[:,2:]-a[:,:2],axis=1)
    bb=np.prod(b[:,2:]-b[:,:2],axis=1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def annotations(record):
    out=[]
    for box in record.get('gtboxes',[]):
        if box.get('tag')!='person' or box.get('extra',{}).get('ignore',0):continue
        full=xyxy(box['fbox']);visible=xyxy(box['vbox'])
        whole=np.prod(full[2:]-full[:2]);shown=np.prod(visible[2:]-visible[:2])
        out.append((full,float(np.clip(1-shown/whole,0,1))))
    return out

def assign_recall(pred,gt,threshold=.5):
    matched=np.zeros(len(gt),bool)
    for box,score in sorted(pred,key=lambda v:-v[1]):
        o=overlap(np.asarray(box)[None,:],[x[0] for x in gt])[0]
        if not len(o):continue
        o[matched]=-1
        j=int(np.argmax(o))
        if o[j]>=threshold:matched[j]=True
    return matched

def main():
    ap=argparse.ArgumentParser()
    for flag in ('image-dir','annotations-odgt','checkpoint','output'):
        ap.add_argument('--'+flag,type=Path,required=True)
    ap.add_argument('--max-images',type=int,default=0)
    ap.add_argument('--score-threshold',type=float,default=.3)
    args=ap.parse_args()
    if not args.image_dir.is_dir():raise FileNotFoundError(args.image_dir)
    for f in (args.annotations_odgt,args.checkpoint):
        if not f.is_file():raise FileNotFoundError(f)
    if args.checkpoint.stat().st_size<1000000:raise ValueError('Real model weights required')
    import torch,torchvision
    model=torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None,weights_backbone=None)
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict' in weights:weights=weights['state_dict']
    model.load_state_dict(weights,strict=True);model.eval()
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'figures').mkdir(exist_ok=True)
    strata={'low':(0,.2),'medium':(.2,.5),'heavy':(.5,1.001)}
    counts={k:[0,0] for k in strata};rows=[];frames=0
    for line in args.annotations_odgt.read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line);file=args.image_dir/(r['ID']+'.jpg')
        if not file.is_file():raise FileNotFoundError(file)
        img=cv2.imread(str(file))
        if img is None:raise ValueError('Invalid original photograph')
        gt=annotations(r)
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy()
        with torch.inference_mode():
            out=model([torch.from_numpy(rgb).permute(2,0,1).float().div(255)])[0]
        bb=out['boxes'].cpu().numpy();sc=out['scores'].cpu().numpy();cl=out['labels'].cpu().numpy()
        pred=[(b,float(s)) for b,s,c in zip(bb,sc,cl) if int(c)==1 and s>=args.score_threshold]
        matched=assign_recall(pred,gt)
        for (box,ratio),hit in zip(gt,matched):
            stratum=next(k for k,(l,h) in strata.items() if l<=ratio<h)
            counts[stratum][0]+=int(hit);counts[stratum][1]+=1
            rows.append([r['ID'],stratum,ratio,int(hit),*map(float,box)])
        if frames<6:
            for box,score in pred:
                x1,y1,x2,y2=map(int,box);cv2.rectangle(img,(x1,y1),(x2,y2),(26,137,190),2)
            for box,ratio in gt:
                x1,y1,x2,y2=map(int,box);cv2.rectangle(img,(x1,y1),(x2,y2),(30,179,104),1)
            cv2.imwrite(str(args.output/'figures'/f'pedestrian_{frames:03d}.png'),img)
        frames+=1
        if args.max_images and frames>=args.max_images:break
    if not frames:raise ValueError('Empty original annotated set')
    with (args.output/'per_person_results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image','occlusion_bin','occlusion_area_proxy','matched_IoU50','x1','y1','x2','y2']);w.writerows(rows)
    with (args.output/'recall_by_occlusion.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['stratum','matched','total','local_recall_IoU50'])
        for k,(good,total) in counts.items():w.writerow([k,good,total,good/total if total else ''])
    info={'model':'torchvision COCO Faster R-CNN ResNet50 FPN person class',
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'dataset':'CrowdHuman original ODGT',
        'evaluation':'Local full-body GT IoU50 recall by visible/full bounding box area ratio; NOT official AP/MR/JI',
        'processed_images':frames,'person_instances':len(rows),'bins':counts}
    (args.output/'run_manifest.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
