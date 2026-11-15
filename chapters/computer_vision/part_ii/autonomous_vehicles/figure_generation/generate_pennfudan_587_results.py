"""Figure 5.51: real Penn-Fudan detector evaluation summary, not new inference.
Source: original 12-image Faster R-CNN experiment, measurements stored in the
manuscript's Table of real Penn-Fudan evaluation results.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
source=out/"pennfudan_archived_summary.csv"
with source.open(newline="") as file:
 d={row["metric"]:float(row["value"]) for row in csv.DictReader(file)}
gt=int(d["reference_persons"]);pred=int(d["person_class_detections"]);tp=int(d["matched_true_positives"])
assert (int(d["annotated_images"]),gt,pred,tp)==(12,21,31,21)
fp=pred-tp;fn=gt-tp;prec=tp/pred;recall=tp/gt;ap50=d["ranked_ap50"]
fig,axs=plt.subplots(1,2,figsize=(10,3.9),layout="constrained")
a=axs[0]
a.bar(["Reference persons","Predictions"],[gt,pred],color=["#506d8f","#09877e"],width=.52)
a.bar(["Predictions"],[fp],bottom=[tp],color="#d4794c",width=.52,label="Unmatched predictions")
a.bar(["Predictions"],[tp],color="#09877e",width=.52,label="Matched predictions")
for x,y in enumerate([gt,pred]):a.text(x,y+.7,str(y),ha="center",fontsize=10)
a.set_ylim(0,37);a.set_ylabel("Object count");a.set_title("(a) Actual evaluated detection counts",fontsize=11)
a.legend(frameon=False,fontsize=8,loc="upper left")
a=axs[1];values=[prec,recall,ap50]
a.barh(["Precision","Recall","Local ranked AP$_{50}$"],values,color=["#506d8f","#09877e","#7a64ad"],height=.5)
a.invert_yaxis();a.set_xlim(0,1.12);a.set_xlabel("Score");a.set_title("(b) Metrics from recorded detections",fontsize=11)
for i,v in enumerate(values):a.text(v+.025,i,f"{v:.4f}",va="center",fontsize=9)
a.grid(axis="x",alpha=.15)
for ext in ("png","pdf","svg"):
 fig.savefig(out/f"pennfudan_recorded_detection_results.{ext}",dpi=210,bbox_inches="tight",facecolor="white")
plt.close(fig)
with (out/"pennfudan_587_derived_metrics.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["metric","value"])
 for k,v in [("tp",tp),("fp",fp),("fn",fn),("precision",prec),("recall",recall),("local_ranked_ap50",ap50)]:w.writerow([k,v])
print("Archived experiment: TP",tp,"FP",fp,"FN",fn,"precision",prec,"recall",recall,"AP50",ap50)
