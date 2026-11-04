"""Run real YOLO+ByteTrack on an original chronological VisDrone VID sequence.
Outputs real track identities and image overlays, not IDF1/HOTA benchmark scores.
"""
import argparse,csv,hashlib,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-frames',type=int,default=100)
    p.add_argument('--confidence',type=float,default=.2)
    a=p.parse_args()
    if not a.checkpoint.is_file() or a.checkpoint.stat().st_size<100000:
        raise FileNotFoundError('Actual YOLO model checkpoint required')
    frames=sorted(p for p in a.frames.glob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not frames:raise FileNotFoundError('Original chronological VisDrone video frames required')
    import cv2
    from ultralytics import YOLO
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'figures').mkdir(exist_ok=True)
    model=YOLO(str(a.checkpoint))
    rows=[]
    for i,p in enumerate(frames[:a.max_frames]):
        img=cv2.imread(str(p))
        if img is None:raise ValueError('Unreadable real frame '+str(p))
        result=model.track(img,persist=True,tracker='bytetrack.yaml',conf=a.confidence,verbose=False)[0]
        overlay=img.copy()
        if result.boxes is not None and len(result.boxes) and result.boxes.id is not None:
            boxes=result.boxes.xyxy.cpu().numpy()
            ids=result.boxes.id.int().cpu().numpy()
            scores=result.boxes.conf.cpu().numpy()
            cls=result.boxes.cls.int().cpu().numpy()
            for box,track_id,score,c in zip(boxes,ids,scores,cls):
                x1,y1,x2,y2=map(int,box)
                rows.append([p.name,i,int(track_id),int(c),float(score),*map(float,box)])
                cv2.rectangle(overlay,(x1,y1),(x2,y2),(12,132,128),2)
                cv2.putText(overlay,f'ID {track_id} {score:.2f}',(x1,max(12,y1-3)),
                   cv2.FONT_HERSHEY_SIMPLEX,.45,(12,132,128),1)
        if i in (0,min(20,len(frames)-1),min(50,len(frames)-1)):
            cv2.imwrite(str(a.output/'figures'/f'track_{i:04d}.png'),overlay)
    with (a.output/'tracks.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['frame','frame_index','track_id','class_id','confidence','x1','y1','x2','y2'])
        w.writerows(rows)
    (a.output/'tracking_manifest.json').write_text(json.dumps({
     'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
     'source_frames':str(a.frames.resolve()),'frames_processed':min(a.max_frames,len(frames)),
     'observations':len(rows),'tracker':'Ultralytics ByteTrack',
     'limitation':'No IDF1/HOTA computed; needs identity ground truth + official evaluator'
    },indent=2)+'\n')
    print('Recorded real-model track observations:',len(rows))
if __name__=='__main__':main()
