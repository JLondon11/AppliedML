#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,platform,sys,time
from dataclasses import asdict,dataclass
from pathlib import Path
import numpy as np,pandas as pd,torch,torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import PILToTensor
import diffusers,torchmetrics
from diffusers import DDPMPipeline,DDPMScheduler,DDIMScheduler,DPMSolverMultistepScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"outputs";OUT.mkdir(parents=True,exist_ok=True)
PALETTE={"DDPM":"#173A52","DDIM":"#66735B","DPM-Solver++":"#725B78"}
MARKERS={"DDPM":"o","DDIM":"s","DPM-Solver++":"^"}
@dataclass
class Environment:
 timestamp_utc:str;platform:str;python:str;torch:str;torchvision:str;diffusers:str;torchmetrics:str
 device_requested:str;device_resolved:str;cuda_available:bool;cuda_version:str|None;cudnn_version:int|None
 gpu_name:str|None;gpu_total_memory_gib:float|None;model_id:str;dataset:str;dataset_split:str
 num_images_per_run:int;batch_size:int;repeats:int;inference_steps:list[int];samplers:list[str];dtype:str
def utc_now():
 from datetime import datetime,timezone
 return datetime.now(timezone.utc).isoformat()
def resolve_device(x):
 if x=="cuda":
  if not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
  return torch.device("cuda")
 if x=="mps":
  if not torch.backends.mps.is_available():raise RuntimeError("MPS requested but unavailable")
  return torch.device("mps")
 if x=="cpu":return torch.device("cpu")
 if torch.cuda.is_available():return torch.device("cuda")
 if torch.backends.mps.is_available():return torch.device("mps")
 return torch.device("cpu")
def scheduler_for(name,cfg):
 if name=="DDPM":return DDPMScheduler.from_config(cfg)
 if name=="DDIM":return DDIMScheduler.from_config(cfg)
 if name=="DPM-Solver++":return DPMSolverMultistepScheduler.from_config(cfg,algorithm_type="dpmsolver++",solver_order=2)
 raise ValueError(name)
def sync(device):
 if device.type=="cuda":torch.cuda.synchronize(device)
 elif device.type=="mps":torch.mps.synchronize()
def real_loader(root,batch,workers):
 ds=CIFAR10(root=root,train=True,download=True,transform=PILToTensor())
 return DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=torch.cuda.is_available())
def add_real(fid,loader,n,device):
 seen=0
 for x,_ in loader:
  if seen>=n:break
  x=x[:n-seen].to(device,non_blocking=True);fid.update(x,real=True);seen+=len(x)
 if seen!=n:raise RuntimeError(f"Expected {n} real images, got {seen}")
def counter(unet):
 s={"calls":0}
 def h(*_):s["calls"]+=1
 return s,unet.register_forward_hook(h)
def fake_uint8(a):return torch.from_numpy(np.clip(a*255,0,255).round().astype(np.uint8)).permute(0,3,1,2).contiguous()
@torch.inference_mode()
def point(pipe,sampler,steps,rep,seed,n,batch,loader,device):
 pipe.scheduler=scheduler_for(sampler,pipe.scheduler.config);pipe.set_progress_bar_config(disable=True)
 _=pipe(batch_size=min(batch,8),generator=torch.Generator(device=device).manual_seed(seed+999999),num_inference_steps=steps,output_type="np");sync(device)
 if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
 fid=FrechetInceptionDistance(feature=2048,normalize=False).to(device);add_real(fid,loader,n,device)
 state,handle=counter(pipe.unet);secs=0.;done=0;batches=0
 try:
  while done<n:
   cur=min(batch,n-done);g=torch.Generator(device=device).manual_seed(seed+batches);sync(device);t=time.perf_counter()
   out=pipe(batch_size=cur,generator=g,num_inference_steps=steps,output_type="np");sync(device);secs+=time.perf_counter()-t
   fid.update(fake_uint8(out.images).to(device,non_blocking=True),real=False);done+=cur;batches+=1
 finally:handle.remove()
 f=float(fid.compute().detach().cpu());calls=state["calls"];nfe=calls/batches
 return {"sampler":sampler,"steps_requested":steps,"repeat":rep,"seed":seed,"num_images":n,"batch_size":batch,
 "fid":f,"generation_seconds":secs,"latency_ms_per_image":1000*secs/n,"throughput_images_per_second":n/secs,
 "unet_forward_calls_total":calls,"nfe_per_sample":nfe,
 "peak_allocated_memory_gib":torch.cuda.max_memory_allocated(device)/(1024**3) if device.type=="cuda" else np.nan}
def dominated(lat,fid):
 d=np.zeros(len(lat),bool)
 for i in range(len(lat)):
  for j in range(len(lat)):
   if i!=j and lat[j]<=lat[i] and fid[j]<=fid[i] and (lat[j]<lat[i] or fid[j]<fid[i]):d[i]=True;break
 return d
def summarize(df):
 s=df.groupby(["sampler","steps_requested"],as_index=False).agg(fid_mean=("fid","mean"),fid_sd=("fid","std"),
 latency_ms_mean=("latency_ms_per_image","mean"),latency_ms_sd=("latency_ms_per_image","std"),
 throughput_mean=("throughput_images_per_second","mean"),throughput_sd=("throughput_images_per_second","std"),
 nfe_mean=("nfe_per_sample","mean"),nfe_sd=("nfe_per_sample","std"),peak_memory_gib_mean=("peak_allocated_memory_gib","mean"),
 repeats=("repeat","count"))
 for c in ["fid_sd","latency_ms_sd","throughput_sd","nfe_sd"]:s[c]=s[c].fillna(0.)
 s["dominated"]=dominated(s.latency_ms_mean.to_numpy(),s.fid_mean.to_numpy());s.to_csv(OUT/"figure25_summary.csv",index=False);return s
def benchmark(a):
 device=resolve_device(a.device);dtype={"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}[a.dtype]
 if dtype==torch.float16 and device.type!="cuda":raise RuntimeError("float16 benchmark supported only on CUDA")
 props=torch.cuda.get_device_properties(device) if device.type=="cuda" else None
 env=Environment(utc_now(),platform.platform(),sys.version.replace("\n"," "),torch.__version__,torchvision.__version__,diffusers.__version__,torchmetrics.__version__,a.device,str(device),torch.cuda.is_available(),torch.version.cuda,torch.backends.cudnn.version() if torch.cuda.is_available() else None,props.name if props else None,props.total_memory/(1024**3) if props else None,a.model_id,"CIFAR-10","train",a.num_images,a.batch_size,a.repeats,list(a.steps),list(a.samplers),str(dtype).replace("torch.",""))
 (OUT/"figure25_environment.json").write_text(json.dumps(asdict(env),indent=2),encoding="utf-8")
 loader=real_loader(Path(a.data_root),a.batch_size,a.workers);pipe=DDPMPipeline.from_pretrained(a.model_id,torch_dtype=dtype).to(device);rows=[]
 for sampler in a.samplers:
  for steps in a.steps:
   for rep in range(a.repeats):
    seed=a.seed+rep*100000+steps*100+a.samplers.index(sampler)
    rows.append(point(pipe,sampler,steps,rep,seed,a.num_images,a.batch_size,loader,device))
    pd.DataFrame(rows).to_csv(OUT/"figure25_runs.csv",index=False)
 summarize(pd.DataFrame(rows))
def plot():
 p=OUT/"figure25_runs.csv"
 if not p.exists():raise SystemExit("No real benchmark data found. Run benchmark first; invented values are prohibited.")
 import matplotlib.pyplot as plt
 s=summarize(pd.read_csv(p));plt.rcParams.update({"font.family":"serif","font.size":9,"axes.labelsize":10,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
 fig,axs=plt.subplots(1,2,figsize=(7.6,3.85),dpi=240);dom=s[s.dominated];nd=s[~s.dominated].sort_values("latency_ms_mean")
 ax=axs[0]
 if len(dom):ax.errorbar(dom.latency_ms_mean,dom.fid_mean,xerr=dom.latency_ms_sd,yerr=dom.fid_sd,fmt="o",ms=4.2,color="0.68",ecolor="0.75",capsize=2,lw=.7,label="Pareto-dominated")
 if len(nd)>1:ax.plot(nd.latency_ms_mean,nd.fid_mean,"-",lw=.9,color="#30343B",alpha=.75)
 for sampler,g in s.groupby("sampler",sort=False):
  g=g[~g.dominated];ax.errorbar(g.latency_ms_mean,g.fid_mean,xerr=g.latency_ms_sd,yerr=g.fid_sd,fmt=MARKERS[sampler],ms=5.2,color=PALETTE[sampler],ecolor=PALETTE[sampler],capsize=2.3,lw=.9,label=sampler)
  for _,r in g.iterrows():ax.annotate(str(int(r.steps_requested)),(r.latency_ms_mean,r.fid_mean),xytext=(4,4),textcoords="offset points",fontsize=6.5,color=PALETTE[sampler])
 ax.set(xlabel="Measured sampling latency (ms/image)",ylabel="FID ↓");ax.grid(True,lw=.4,alpha=.22);ax.legend(frameon=False,fontsize=7)
 ax=axs[1]
 if len(dom):ax.errorbar(dom.nfe_mean,dom.fid_mean,xerr=dom.nfe_sd,yerr=dom.fid_sd,fmt="o",ms=4.2,color="0.68",ecolor="0.75",capsize=2,lw=.7)
 for sampler,g in s.groupby("sampler",sort=False):
  g=g[~g.dominated].sort_values("nfe_mean")
  if len(g)>1:ax.plot(g.nfe_mean,g.fid_mean,"-",lw=.8,color=PALETTE[sampler],alpha=.55)
  ax.errorbar(g.nfe_mean,g.fid_mean,xerr=g.nfe_sd,yerr=g.fid_sd,fmt=MARKERS[sampler],ms=5.2,color=PALETTE[sampler],ecolor=PALETTE[sampler],capsize=2.3,lw=.9)
 ax.set(xlabel="Measured UNet function evaluations per sample",ylabel="FID ↓");ax.grid(True,lw=.4,alpha=.22)
 for ax in axs:
  for sp in ax.spines.values():sp.set_linewidth(.55)
 fig.tight_layout();fig.savefig(OUT/"figure25.svg",bbox_inches="tight");fig.savefig(OUT/"figure25.pdf",bbox_inches="tight");fig.savefig(OUT/"figure25.png",dpi=300,bbox_inches="tight");plt.close(fig)
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
 b=sub.add_parser("benchmark");b.add_argument("--model-id",default="google/ddpm-cifar10-32");b.add_argument("--data-root",default=str(HERE.parent/"data"));b.add_argument("--device",choices=["auto","cuda","mps","cpu"],default="auto");b.add_argument("--dtype",choices=["float32","float16","bfloat16"],default="float32");b.add_argument("--num-images",type=int,default=5000);b.add_argument("--batch-size",type=int,default=64);b.add_argument("--workers",type=int,default=4);b.add_argument("--repeats",type=int,default=3);b.add_argument("--seed",type=int,default=25025);b.add_argument("--steps",type=int,nargs="+",default=[10,20,50,100]);b.add_argument("--samplers",nargs="+",choices=["DDPM","DDIM","DPM-Solver++"],default=["DDPM","DDIM","DPM-Solver++"]);sub.add_parser("plot");a=p.parse_args();benchmark(a) if a.command=="benchmark" else plot()
if __name__=="__main__":main()
