"""TuSimple lane-mask inference using a lane-specific single-output ONNX model.
LOCAL pixel metrics, not official TuSimple lane challenge accuracy.
Require real image files, official JSONL annotations, and compatible checkpoint.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import cv2

def truth_mask(record,shape,thickness=4):
    h,w=shape;mask=np.zeros((h,w),np.uint8)
    ys=record['h_samples']
    for lane in record['lanes']:
        if len(lane)!=len(ys):raise ValueError('Mismatched lane and h_samples')
        pts=[(int(x),int(y)) for x,y in zip(lane,ys) if int(x)>=0 and 0<=int(y)<h]
        if len(pts)>1:cv2.polylines(mask,[np.array(pts,np.int32)],False,1,thickness)
    return mask

def extract_binary_mask(outputs,shape,threshold=.5):
    if len(outputs)!=1:raise ValueError('One output required')
    a=np.asarray(outputs[0])
    if a.ndim!=4 or a.shape[:2]!=(1,1):raise ValueError('Expected 1x1xHxW')
    a=a[0,0].astype(np.float32)
    if not np.isfinite(a).all():raise ValueError('Nonfinite output')
    if a.min()<0 or a.max()>1:a=1/(1+np.exp(-np.clip(a,-50,50)))
    return (cv2.resize(a,shape[::-1])>=threshold).astype(np.uint8)

def score(mask,gt):
    m=np.asarray(mask,bool);g=np.asarray(gt,bool)
    if m.shape!=g.shape:raise ValueError('Shape mismatch')
    tp=int(np.sum(m&g));fp=int(np.sum(m&~g));fn=int(np.sum(~m&g))
    return tp,fp,fn,tp/max(1,tp+fp),tp/max(1,tp+fn),tp/max(1,tp+fp+fn)

def main():
    p=argparse.ArgumentParser()
    for key in ('image-root','annotation-jsonl','checkpoint-onnx','output'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--max-images',type=int,default=0)
    p.add_argument('--mask-threshold',type=float,default=.5)
    a=p.parse_args()
    if not a.image_root.is_dir():raise FileNotFoundError(a.image_root)
    for file in (a.annotation_jsonl,a.checkpoint_onnx):
        if not file.is_file():raise FileNotFoundError(file)
    if a.checkpoint_onnx.stat().st_size<100000:raise ValueError('Checkpoint file suspiciously small')
    import onnxruntime as ort
    sess=ort.InferenceSession(str(a.checkpoint_onnx),providers=['CPUExecutionProvider'])
    inp=sess.get_inputs()
    if len(inp)!=1 or len(sess.get_outputs())!=1:raise ValueError('Single input/output required')
    shape=inp[0].shape
    if len(shape)!=4 or not all(isinstance(v,int) and v>0 for v in shape[2:]):
        raise ValueError('Fixed BCHW image size required')
    ih,iw=shape[2:]
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'figures').mkdir(exist_ok=True)
    rows=[];tot=np.zeros(3,dtype=np.int64)
    for line in a.annotation_jsonl.read_text().splitlines():
        if not line.strip():continue
        sample=json.loads(line);file=a.image_root/sample['raw_file']
        if not file.is_file():raise FileNotFoundError(file)
        image=cv2.imread(str(file))
        if image is None:raise ValueError('Unreadable original JPEG')
        gt=truth_mask(sample,image.shape[:2])
        input_array=cv2.cvtColor(cv2.resize(image,(iw,ih)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        input_array=np.transpose(input_array,(2,0,1))[None]
        mask=extract_binary_mask(sess.run(None,{inp[0].name:input_array}),
                                 image.shape[:2],a.mask_threshold)
        stats=score(mask,gt);tot+=stats[:3]
        rows.append([sample['raw_file'],*stats])
        if len(rows)<=8:
            overlay=image.copy()
            overlay[gt.astype(bool)]=np.array([30,165,30])
            overlay[mask.astype(bool)]=np.array([50,90,230])
            cv2.imwrite(str(a.output/'figures'/f'lane_{len(rows):03d}.png'),overlay)
        if a.max_images and len(rows)>=a.max_images:break
    if not rows:raise ValueError('Empty real dataset')
    with (a.output/'per_image_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image','TP','FP','FN','precision','recall','IoU']);w.writerows(rows)
    tp,fp,fn=map(int,tot)
    report={'images':len(rows),'micro_pixel_precision':tp/max(1,tp+fp),
       'micro_pixel_recall':tp/max(1,tp+fn),'micro_pixel_iou':tp/max(1,tp+fp+fn),
       'checkpoint_sha256':hashlib.sha256(a.checkpoint_onnx.read_bytes()).hexdigest(),
       'evaluation':'LOCAL 4-pixel polyline raster metric, NOT TuSimple leaderboard score',
       'input_preprocessing':'RGB /255; validate compatibility with model training'}
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
