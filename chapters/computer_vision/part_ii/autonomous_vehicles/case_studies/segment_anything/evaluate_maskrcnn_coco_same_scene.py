#!/usr/bin/env python3
"""Real COCO val2017 Mask R-CNN ResNet50-FPN automatic mask inference.

A supervised, COCO-pretrained, prompt-free comparator on the EXACT same images
and eligible original instance IDs as the seeded SAM ViT-B validation.
NOT COCO AP or official segmentation benchmark metrics: each GT instance's
mask-IoU is set to 0 if no same-class prediction box matches at IoU>=0.5.
A detector is NOT given oracle ground-truth box prompts, unlike SAM.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def bbox_iou(a,b):
    low=np.maximum(np.asarray(a[:2],float),np.asarray(b[:2],float))
    high=np.minimum(np.asarray(a[2:],float),np.asarray(b[2:],float))
    area=np.prod(np.maximum(high-low,0))
    aa=np.prod(np.maximum(np.asarray(a[2:])-np.asarray(a[:2]),0))
    bb=np.prod(np.maximum(np.asarray(b[2:])-np.asarray(b[:2]),0))
    return float(area/max(float(aa+bb-area),1e-12))

def mask_iou(a,b):
    a=np.asarray(a,bool);b=np.asarray(b,bool)
    if a.shape!=b.shape:raise ValueError('mismatched masks')
    inter=np.count_nonzero(a&b);union=np.count_nonzero(a|b)
    return float(inter/union) if union else 1.

def class_match(pred_boxes,pred_labels,pred_scores,gt_box,gt_cat,used,score_min=.05):
    """Greedy highest-confidence GT-category prediction with box IoU >= 0.5."""
    order=np.argsort(-np.asarray(pred_scores))
    for j in order:
        if j in used or pred_scores[j]<score_min or int(pred_labels[j])!=int(gt_cat):
            continue
        if bbox_iou(pred_boxes[j],gt_box)>=.5:
            used.add(int(j))
            return int(j)
    return None

def plot(image,gt,pred,box,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.7),facecolor='white')
    for k,(ax,mask) in enumerate(zip(axes,[None,gt,pred])):
        ax.imshow(image);ax.axis('off')
        if mask is not None:
            ax.imshow(np.ma.masked_where(~mask,mask),cmap='summer',alpha=.46,vmin=0,vmax=1)
        if k==0:
            x,y,x2,y2=box;ax.add_patch(Rectangle((x,y),x2-x,y2-y,fill=False,
                edgecolor='#c47840',lw=1.5))
        ax.text(.5,-.065,'('+'abc'[k]+')',transform=ax.transAxes,ha='center')
    fig.subplots_adjust(left=.01,right=.99,bottom=.12,top=.98,wspace=.03)
    for ext in ('png','svg','pdf'):
        fig.savefig(out.with_suffix('.'+ext),dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    for key in ('coco-json','images','output'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--max-objects',type=int,default=64)
    p.add_argument('--max-plots',type=int,default=6)
    p.add_argument('--score-min',type=float,default=.05)
    p.add_argument('--min-box-area',type=float,default=3000)
    a=p.parse_args()
    if not a.coco_json.is_file() or not a.images.is_dir():
        raise FileNotFoundError('Actual COCO image subset and original labels required')
    import torch,torchvision
    from pycocotools.coco import COCO
    from torchvision.models.detection import maskrcnn_resnet50_fpn,MaskRCNN_ResNet50_FPN_Weights
    torch.set_num_threads(2)
    weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model=maskrcnn_resnet50_fpn(weights=weights).eval()
    cache=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
    if not cache.is_file():raise FileNotFoundError('Actual COCO pretrained weights did not load')
    coco=COCO(str(a.coco_json))
    names={cat['id']:cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'Figures').mkdir(exist_ok=True)
    rows=[];nimgs=0;ndets=0
    for image_id in sorted(coco.getImgIds()):
        info=coco.loadImgs([image_id])[0]
        file=a.images/info['file_name']
        if not file.is_file():raise FileNotFoundError(file)
        img=Image.open(file).convert('RGB')
        tensor=weights.transforms()(img)
        with torch.inference_mode():
            out=model([tensor])[0]
        boxes=out['boxes'].cpu().numpy()
        labels=out['labels'].cpu().numpy()
        scores=out['scores'].cpu().numpy()
        masks=out['masks'][:,0].cpu().numpy()>=.5
        ndets+=int(sum(scores>=a.score_min));nimgs+=1
        used=set()
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id],iscrowd=False)):
            if ann.get('area',0)<a.min_box_area:continue
            gt=coco.annToMask(ann).astype(bool)
            if not gt.any():continue
            x,y,w,h=map(float,ann['bbox'])
            idx=class_match(boxes,labels,scores,[x,y,x+w,y+h],ann['category_id'],used,a.score_min)
            imask=masks[idx] if idx is not None else np.zeros_like(gt,bool)
            iou=mask_iou(imask,gt)
            rows.append([image_id,ann['id'],names[ann['category_id']],float(ann['area']),
                         idx is not None, float(scores[idx]) if idx is not None else '',
                         iou, float(bbox_iou(boxes[idx],[x,y,x+w,y+h])) if idx is not None else ''])
            if len(rows)<=a.max_plots:
                plot(img,gt,imask,(x,y,x+w,y+h),a.output/'Figures'/f'maskrcnn_{image_id}_{ann["id"]}')
            if len(rows)>=a.max_objects:break
        if len(rows)>=a.max_objects:break
    if not rows:raise ValueError('No real labeled COCO objects found')
    with (a.output/'per_object_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['image_id','annotation_id','class','annotation_area',
              'matched','confidence','mask_iou','matched_box_iou']);writer.writerows(rows)
    summary=dict(images=nimgs,objects=len(rows),matched=sum(int(x[4]) for x in rows),
                 mean_object_iou=float(np.mean([x[6] for x in rows])),
                 object_match_fraction=float(np.mean([x[4] for x in rows])),
                 predicted_candidates_at_score_cutoff=ndets,
                 checkpoint_sha256=sha256(cache),annotations_sha256=sha256(a.coco_json),
                 model='Official torchvision COCO pretrained Mask R-CNN ResNet50 FPN',
                 checkpoint_url=weights.url,torch_version=str(torch.__version__),
                 torchvision_version=str(torchvision.__version__),
                 protocol='Class/box IoU50 matched automatic instance masks; unmatched GT IoU=0; no oracle prompt',
                 limitations='Not COCO AP; selected nonrandom sample; baseline is not prompt-matched to SAM')
    (a.output/'metrics_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
