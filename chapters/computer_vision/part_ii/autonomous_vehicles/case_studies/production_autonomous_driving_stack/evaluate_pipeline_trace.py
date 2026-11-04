"""Evaluate genuine recorded perception/tracking/planning telemetry, not safety.

CSV: frame_id,timestamp_s,perception_ms,tracking_ms,planning_ms,
obstacle_count,track_count,ego_speed_mps,planner_valid,emergency_brake
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
FIELDS=('frame_id','timestamp_s','perception_ms','tracking_ms','planning_ms',
  'obstacle_count','track_count','ego_speed_mps','planner_valid','emergency_brake')
def load_trace(path):
    with open(path,newline='') as f:
        rd=csv.DictReader(f)
        if not set(FIELDS).issubset(rd.fieldnames or []):raise ValueError('Missing actual telemetry schema')
        rows=list(rd)
    if len(rows)<20:raise ValueError('At least 20 original successive frames required')
    if len({r['frame_id'] for r in rows})!=len(rows):raise ValueError('Repeated frame ID')
    vals={c:np.array([float(r[c]) for r in rows]) for c in FIELDS if c!='frame_id'}
    if any(not np.isfinite(a).all() for a in vals.values()):raise ValueError('Nonfinite telemetry')
    if np.any(np.diff(vals['timestamp_s'])<=0):raise ValueError('Nonmonotonic timestamps')
    for c in ('perception_ms','tracking_ms','planning_ms','obstacle_count','track_count','ego_speed_mps'):
        if (vals[c]<0).any():raise ValueError('Invalid negative telemetry value')
    for c in ('planner_valid','emergency_brake'):
        if not np.isin(vals[c],[0,1]).all():raise ValueError('Invalid Boolean flag')
    return vals
def summarize(v,deadline_ms):
    t=v['perception_ms']+v['tracking_ms']+v['planning_ms']
    return dict(frames=len(t),
      recorded_duration_s=float(v['timestamp_s'][-1]-v['timestamp_s'][0]),
      latency_median_ms=float(np.median(t)),latency_p95_ms=float(np.percentile(t,95)),
      latency_p99_ms=float(np.percentile(t,99)),
      deadline_miss_fraction=float(np.mean(t>deadline_ms)),
      planner_valid_fraction=float(np.mean(v['planner_valid'])),
      emergency_brake_command_fraction=float(np.mean(v['emergency_brake'])),
      mean_detections=float(np.mean(v['obstacle_count'])),
      mean_tracks=float(np.mean(v['track_count'])))
def make_plot(v,deadline_ms,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt=v['timestamp_s']-v['timestamp_s'][0]
    total=v['perception_ms']+v['tracking_ms']+v['planning_ms']
    fig,axs=plt.subplots(1,3,figsize=(11,3.3),facecolor='white')
    axs[0].plot(dt,total,color='#167b78',lw=1)
    axs[0].axhline(deadline_ms,color='#c56e40',ls='--')
    axs[0].set(xlabel='Recorded elapsed time (s)',ylabel='Measured latency (ms)')
    axs[1].plot(dt,v['obstacle_count'],label='Detections',color='#526b85')
    axs[1].plot(dt,v['track_count'],label='Tracks',color='#c56e40')
    axs[1].legend(fontsize=7);axs[1].set(xlabel='Recorded time (s)',ylabel='Count/frame')
    axs[2].bar(['Invalid plan','Brake command'],[
      np.sum(v['planner_valid']==0),np.sum(v['emergency_brake']==1)],
      color=['#c56e40','#167b78'])
    axs[2].set_ylabel('Recorded frames')
    for i,ax in enumerate(axs):
        ax.spines[['top','right']].set_visible(False)
        ax.text(.5,-.25,'('+'abc'[i]+')',ha='center',transform=ax.transAxes,color='black')
    fig.subplots_adjust(left=.08,right=.99,bottom=.29,top=.96,wspace=.32)
    for ext in ('pdf','png','svg'):fig.savefig(target.with_suffix('.'+ext),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-ms',type=float,default=100)
    a=p.parse_args()
    if a.deadline_ms<=0:raise ValueError('Deadline must be positive')
    if not a.trace.is_file():raise FileNotFoundError('Original recorded telemetry trace required')
    v=load_trace(a.trace);report=summarize(v,a.deadline_ms)
    a.output.mkdir(parents=True,exist_ok=True)
    make_plot(v,a.deadline_ms,a.output/'real_pipeline_trace')
    with (a.output/'metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','value']);w.writerows(report.items())
    info=dict(trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
      deadline_ms=a.deadline_ms,source='user-supplied original recorded telemetry',
      caveat='No detector accuracy, tracking ground truth, vehicle crash metric or road safety evidence')
    (a.output/'provenance.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
