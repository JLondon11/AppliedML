"""Evaluate original Meta SAM checkpoints on real COCO val2017 instance masks.

Box prompts are COCO ground-truth boxes (oracle prompts). This local per-instance
mask IoU/Dice protocol is NOT COCO AP or the original SAM paper's benchmark.
Requires original COCO images/annotations and actual original SAM checkpoint.
"""
import argparse,csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def xywh_to_xyxy(box,w,h):
    x,y,bw,bh=map(float,box)
    if not np.isfinite([x,y,bw,bh]).all() or bw<=0 or bh<=0:raise ValueError('Bad COCO box')
    b=np.asarray([max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)],np.float32)
    if b[2]<=b[0] or b[3]<=b[1]:raise ValueError('Box outside source image')
    return b

def mask_scores(pred,groundtruth):
    a=np.asarray(pred,bool);b=np.asarray(groundtruth,bool)
    if a.ndim!=2 or a.shape!=b.shape:raise ValueError('Mask shape mismatch')
    inter=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b))
    denom=int(a.sum()+b.sum())
    return inter/union if union else 1.,2*inter/denom if denom else 1.

def grabcut_box_baseline(bgr,bbox,iterations=5):
    import cv2
    h,w=bgr.shape[:2];x1,y1,x2,y2=map(lambda v:int(round(v)),bbox)
    x1=max(0,min(x1,w-2));y1=max(0,min(y1,h-2))
    x2=min(w-1,max(x2,x1+2));y2=min(h-1,max(y2,y1+2))
    if x1==0 or y1==0 or x2>=w-1 or y2>=h-1:
        raise ValueError('GrabCut rectangle too close to image border')
    mask=np.zeros((h,w),np.uint8);bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
    cv2.grabCut(bgr,mask,(x1,y1,x2-x1,y2-y1),bg,fg,int(iterations),cv2.GC_INIT_WITH_RECT)
    return np.isin(mask,[cv2.GC_FGD,cv2.GC_PR_FGD])

def render_panels(image,gt,sam_mask,grabcut_mask,box,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,4,figsize=(11.5,3.5),facecolor='white')
    for j,(ax,mask) in enumerate(zip(axes,[None,gt,sam_mask,grabcut_mask])):
        ax.imshow(image);ax.axis('off')
        if mask is not None:
            overlay=np.ma.masked_where(~np.asarray(mask,bool),np.asarray(mask,float))
            ax.imshow(overlay,cmap='summer',alpha=.43,vmin=0,vmax=1)
            ax.contour(np.asarray(mask,float),levels=[.5],colors=['#187a76'],linewidths=.7)
        if j==0:ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                                   fill=False,edgecolor='#c36c31',lw=1.5))
        ax.text(.5,-.065,'('+'abcd'[j]+')',transform=ax.transAxes,ha='center',color='black')
    fig.subplots_adjust(left=.01,right=.99,bottom=.12,top=.98,wspace=.03)
    for ext in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+ext),dpi=190,facecolor='white',
                    bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    for name in ['coco-json','images','checkpoint','output']:
        ap.add_argument('--'+name,required=True,type=Path)
    ap.add_argument('--model-type',choices=['vit_b','vit_l','vit_h'],default='vit_b')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--max-objects',type=int,default=80)
    ap.add_argument('--max-plots',type=int,default=6)
    ap.add_argument('--min-box-area',type=float,default=3000)
    a=ap.parse_args()
    if not a.coco_json.is_file() or not a.images.is_dir():
        raise FileNotFoundError('Original COCO val2017 photos and instances annotations required')
    if not a.checkpoint.is_file() or a.checkpoint.stat().st_size<10_000_000:
        raise FileNotFoundError('Actual Meta SAM original checkpoint required')
    if a.max_objects<1:raise ValueError('max-objects must be positive')
    import cv2,torch,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pycocotools.coco import COCO
    from segment_anything import sam_model_registry,SamPredictor
    ds=COCO(str(a.coco_json))
    network=sam_model_registry[a.model_type](checkpoint=str(a.checkpoint)).to(device=a.device)
    network.eval()
    predictor=SamPredictor(network)
    names={v['id']:v['name'] for v in ds.loadCats(ds.getCatIds())}
    a.output.mkdir(parents=True,exist_ok=True)
    figs=a.output/'Figures';figs.mkdir(exist_ok=True)
    rows=[]
    for id in sorted(ds.getImgIds()):
        meta=ds.loadImgs([id])[0];file=a.images/meta['file_name']
        if not file.is_file():raise FileNotFoundError(file)
        bgr=cv2.imread(str(file))
        if bgr is None:raise ValueError('Unreadable real COCO photograph')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
        eligible=[]
        for ann in ds.loadAnns(ds.getAnnIds(imgIds=[id],iscrowd=False)):
            if ann.get('area',0)<a.min_box_area:continue
            try:
                box=xywh_to_xyxy(ann['bbox'],w,h)
                gt=ds.annToMask(ann).astype(bool)
                if gt.any():eligible.append((ann,box,gt))
            except ValueError:continue
        if not eligible:continue
        predictor.set_image(rgb)
        for ann,box,gt in eligible:
            with torch.inference_mode():
                masks,quality,_=predictor.predict(box=box,multimask_output=False)
            if masks.shape!=(1,h,w):raise ValueError('Unexpected SAM result shape')
            pred=np.asarray(masks[0],bool);score=mask_scores(pred,gt)
            try:
                baseline=grabcut_box_baseline(bgr,box);bs=mask_scores(baseline,gt)
            except (ValueError,cv2.error):
                baseline=None;bs=(float('nan'),float('nan'))
            name=names.get(ann['category_id'],'unknown')
            rows.append([id,ann['id'],name,int(gt.sum()),*score,*bs,
                         float(quality[0]),bool(baseline is not None)])
            if len(rows)<=a.max_plots:
                render_panels(rgb,gt,pred,baseline,box,
                              figs/f'sam_coco_{id}_{ann["id"]}')
            if len(rows)>=a.max_objects:break
        if len(rows)>=a.max_objects:break
    if not rows:raise RuntimeError('No valid COCO objects; no evaluation recorded')
    with (a.output/'per_object_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['image_id','annotation_id','class',
           'ground_truth_area_px','sam_iou','sam_dice','grabcut_iou',
           'grabcut_dice','sam_predicted_iou','grabcut_valid']);writer.writerows(rows)
    groups=defaultdict(list)
    for r in rows:groups[r[2]].append(r)
    with (a.output/'per_class_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['class','instances','sam_mean_iou',
                         'grabcut_mean_iou_paired','sam_mean_iou_paired','valid_pairs'])
        for name,samples in sorted(groups.items()):
            paired=[r for r in samples if r[9]]
            writer.writerow([name,len(samples),float(np.mean([r[4] for r in samples])),
                    float(np.mean([r[6] for r in paired])) if paired else '',
                    float(np.mean([r[4] for r in paired])) if paired else '',len(paired)])
    pairs=[r for r in rows if r[9]]
    fig,ax=plt.subplots(figsize=(6,3.5),facecolor='white')
    if pairs:ax.scatter([r[6] for r in pairs],[r[4] for r in pairs],
                         color='#16817f',alpha=.7,s=18)
    ax.plot([0,1],[0,1],'--',color='#7c8695',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='GrabCut mask IoU (same GT-box prompt)',
           ylabel='SAM mask IoU (GT-box prompt)')
    fig.tight_layout()
    for ext in ['png','pdf','svg']:
        fig.savefig(figs/f'sam_vs_grabcut_iou.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
    report={'dataset':'COCO 2017 validation instance masks',
        'model':f'Meta SAM {a.model_type}',
        'prompt_protocol':'Oracle COCO ground-truth box, no fine-tuning',
        'instances':len(rows),'images':len(set(r[0] for r in rows)),
        'sam_mean_iou':float(np.mean([r[4] for r in rows])),
        'sam_mean_dice':float(np.mean([r[5] for r in rows])),
        'grabcut_paired_samples':len(pairs),
        'paired_sam_iou':float(np.mean([r[4] for r in pairs])) if pairs else None,
        'paired_grabcut_iou':float(np.mean([r[6] for r in pairs])) if pairs else None,
        'checkpoint_sha256':sha256_file(a.checkpoint),
        'annotations_sha256':sha256_file(a.coco_json),
        'torch_version':str(torch.__version__),
        'warning':'Not COCO AP, not prompt-free, not official SAM benchmark.'}
    (a.output/'metrics_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
