"""Visualize actual OpenPCDet result.pkl predictions over their KITTI LiDAR frames.

Must read an OpenPCDet model output file and real KITTI .bin data. No fake detections.
This figure is qualitative; use OpenPCDet official evaluator for quantitative AP.
"""
import argparse
import csv
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def corners_bev(box):
    """OpenPCDet LiDAR boxes: (x,y,z,dx,dy,dz,yaw)."""
    x,y,_,dx,dy,_,yaw=box[:7]
    local=np.array([[-dx/2,-dy/2],[dx/2,-dy/2],[dx/2,dy/2],[-dx/2,dy/2],[-dx/2,-dy/2]])
    rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    return local@rot.T+np.array([x,y])


def parse_pred(obj):
    if not isinstance(obj,dict):raise ValueError('Expected OpenPCDet prediction dictionary')
    if 'boxes_lidar' not in obj:
        raise ValueError('No boxes_lidar in result: cannot verify BEV geometry. Do not invent it.')
    boxes=np.asarray(obj['boxes_lidar'],dtype=float)
    if boxes.size==0:boxes=np.empty((0,7),dtype=float)
    if boxes.ndim!=2 or boxes.shape[1]<7:raise ValueError('Invalid LiDAR boxes shape')
    if not np.isfinite(boxes).all():raise ValueError('Nonfinite prediction geometry')
    scores=np.asarray(obj.get('score',[]),dtype=float)
    if len(scores)!=len(boxes):raise ValueError('Mismatched predictions and scores')
    names=list(obj.get('name',[]))
    if len(names)!=len(boxes):raise ValueError('Mismatched predictions and names')
    frame=str(obj.get('frame_id',''))
    if frame.isdigit():frame=frame.zfill(6)
    if not frame or not frame.isdigit():raise ValueError('Invalid KITTI frame id')
    return frame,boxes,scores,names


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--result-pkl',type=Path,required=True)
    p.add_argument('--kitti',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--score-threshold',type=float,default=0.35)
    p.add_argument('--max-frames',type=int,default=6)
    a=p.parse_args()
    if not (0<=a.score_threshold<=1):raise ValueError('Score threshold outside [0,1]')
    if a.max_frames<1:raise ValueError('At least one frame required')
    with open(a.result_pkl,'rb') as f: predictions=pickle.load(f)
    if not isinstance(predictions,list) or not predictions:
        raise ValueError('OpenPCDet results must be a nonempty list; no figure generated')
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for item in predictions[:a.max_frames]:
        frame,boxes,scores,names=parse_pred(item)
        cloud=a.kitti/'training'/'velodyne'/f'{frame}.bin'
        if not cloud.is_file():raise FileNotFoundError(f'Missing original KITTI frame {cloud}')
        points=np.fromfile(cloud,dtype=np.float32)
        if len(points)%4:raise ValueError('Invalid 4-float KITTI LiDAR file')
        points=points.reshape(-1,4)
        keep=(points[:,0]>0)&(points[:,0]<70.4)&(abs(points[:,1])<40)
        fig,ax=plt.subplots(figsize=(10.3,5.2),facecolor='white')
        ax.scatter(points[keep,0],points[keep,1],s=.18,c='#526b7d',alpha=.40,rasterized=True)
        n=0
        for b,s,name in zip(boxes,scores,names):
            if s<a.score_threshold:continue
            polygon=corners_bev(b)
            ax.plot(polygon[:,0],polygon[:,1],lw=1.5,color='#e48134')
            ax.text(b[0],b[1]+.45,f'{name} {s:.2f}',fontsize=6,color='#21445a')
            rows.append([frame,name,float(s),*[float(v) for v in b[:7]]])
            n+=1
        ax.set(xlim=(0,70.4),ylim=(-40,40),xlabel='Forward distance x (m)',
               ylabel='Leftward distance y (m)')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.1)
        ax.text(.01,.98,'PointPillars model outputs on KITTI frame '+frame,
                transform=ax.transAxes,va='top',fontsize=9,color='#213445')
        fig.tight_layout()
        fig.savefig(a.output/f'pointpillars_{frame}.png',dpi=190,bbox_inches='tight')
        fig.savefig(a.output/f'pointpillars_{frame}.pdf',bbox_inches='tight')
        plt.close(fig)
    with (a.output/'predictions_displayed.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['frame','class','score','x','y','z','dx','dy','dz','yaw']);w.writerows(rows)
    print('Visualized',min(a.max_frames,len(predictions)),'real KITTI frames with',len(rows),'model predictions')


if __name__=='__main__':main()
