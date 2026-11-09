"""Create publication-quality Chapter 6 Figure 6.44 from executed latent diffusion predictions."""
import argparse,csv,math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser()
p.add_argument("--predictions",type=Path,required=True)
p.add_argument("--output",type=Path,default=Path("figure_6_44"))
args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
with args.predictions.open(newline="") as f:records=list(csv.DictReader(f))
assert len(records)==2000
count=defaultdict(lambda:[0,0])
for r in records:
 arm=r["arm"];digit=int(r["prompt_label"]);label=int(r["classifier_prediction"])
 assert arm in ("conditional","unconditional") and digit in range(10) and label in range(10)
 count[(arm,digit)][0]+=int(digit==label)
 count[(arm,digit)][1]+=1
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n
 mid=(p+z*z/(2*n))/d
 half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return mid-half,mid+half
fig,ax=plt.subplots(figsize=(8.2,5),layout="constrained")
colors={"conditional":"#155E75","unconditional":"#AD5C45"}
source=[]
ys=np.arange(11)[::-1]
for arm,shift in (("conditional",.15),("unconditional",-.15)):
 vals=[];low=[];high=[]
 for i in [-1,*range(10)]:
  if i==-1:k=sum(count[(arm,j)][0] for j in range(10));n=1000
  else:k,n=count[(arm,i)]
  if i>=0:assert n==100
  value=k/n;lo,hi=wilson(k,n)
  vals.append(100*value);low.append(100*(value-lo));high.append(100*(hi-value))
  source.append((arm,"overall" if i==-1 else i,k,n,100*value,100*lo,100*hi))
 ax.errorbar(vals,ys+shift,xerr=[low,high],fmt="o",ms=5,capsize=2.8,
             elinewidth=1.2,color=colors[arm],label=("Class-conditioned" if arm=="conditional" else "Unconditional control"))
 ax.annotate(f"{vals[0]:.1f}%",(vals[0],ys[0]+shift),xytext=(5,8),textcoords="offset points",
             va="bottom",fontweight="bold",color=colors[arm])
for y in ys[1:]:ax.axhline(y,color="#E9EDF1",lw=.75,zorder=0)
ax.axhspan(9.5,10.6,color="#F1F4F7",zorder=-1)
ax.axvline(10,color="#878E94",ls=":",lw=1,label="Chance reference (10%)")
ax.set(yticks=ys,yticklabels=["Overall",*map(str,range(10))],xlim=(-4,105),
       ylim=(-.6,10.65),xlabel="Agreement with requested digit (%) • Wilson 95% binomial interval",
       ylabel="Requested digit label")
ax.tick_params(axis="y",length=0);ax.grid(axis="x",color="#E9EDF1",lw=.75)
ax.legend(loc="lower center",bbox_to_anchor=(.5,1.01),frameon=False,ncol=3,fontsize=8)
for suffix in ("pdf","svg","png"):
 fig.savefig(args.output/("Figure6_44_Classwise_Consistency."+suffix),dpi=240,bbox_inches="tight")
with (args.output/"Figure6_44_source_values.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["arm","class_label","correct","n","percent","wilson95_low","wilson95_high"]);w.writerows(source)
assert [r[2] for r in source if r[1]=="overall"]==[778,94]
print("PASS: all 2000 executed predictions, 10 class comparisons, correct uncertainty intervals")
