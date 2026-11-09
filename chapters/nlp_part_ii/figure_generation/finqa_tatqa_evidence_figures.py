#!/usr/bin/env python3
"""Publication-quality real-data figure replacements for Chapter 9.

Reads original FinQA and TAT-QA official held-out JSON. Images characterize
gold annotations, NOT model predictions or enterprise authorization.
No numerical values are hardcoded or simulated.
"""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--finqa-test",type=Path,required=True)
 p.add_argument("--tatqa-test",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 finqa=json.loads(a.finqa_test.read_text());tatqa=json.loads(a.tatqa_test.read_text())
 assert len(finqa)==1147 and len(tatqa)==277
 counts=[];cat=Counter()
 for item in finqa:
  keys=list(item["qa"]["gold_inds"])
  counts.append(len(keys))
  types={key.split("_")[0] for key in keys}
  cat["Both" if {"table","text"}<=types else
      "Text only" if types=={"text"} else
      "Table only" if types=={"table"} else "Other"]+=1
 qa=[q for d in tatqa for q in d["questions"]]
 assert len(qa)==1663
 classes=["arithmetic","span","multi-span","count"]
 sources=["table","table-text","text"]
 matrix=np.array([[sum(q["answer_type"]==t and q["answer_from"]==s for q in qa)
                   for s in sources] for t in classes])
 assert matrix.sum()==1663
 navy="#18324f";steel="#517c96";teal="#187d85";gray="#7b8791"
 plt.rcParams.update({"font.size":10.5,"font.family":"DejaVu Sans",
                      "axes.spines.top":False,"axes.spines.right":False})
 fig,axes=plt.subplots(1,2,figsize=(11.8,4.9),gridspec_kw={"width_ratios":[1.15,1]})
 labels=["Table only","Text only","Both","Other"];values=[cat[x] for x in labels]
 axes[0].barh(labels[::-1],values[::-1],color=[gray,teal,steel,navy][::-1],height=.58)
 for i,v in enumerate(values[::-1]):axes[0].text(v+8,i,str(v),va="center",color=navy)
 axes[0].set_xlim(0,max(values)*1.17);axes[0].set_xlabel("Official test questions (count)")
 axes[0].set_title("Gold evidence modality",loc="left",fontweight="bold",color=navy)
 bins=[sum(x==i for x in counts) for i in range(1,5)]+[sum(x>=5 for x in counts)]
 axes[1].bar(["1","2","3","4","≥5"],bins,color=steel,width=.65)
 for i,v in enumerate(bins):axes[1].text(i,v+12,str(v),ha="center",color=navy)
 axes[1].set_ylim(0,max(bins)*1.13);axes[1].set_xlabel("Gold evidence items per question")
 axes[1].set_ylabel("Questions");axes[1].set_title("Evidence demand per question",
                                             loc="left",fontweight="bold",color=navy)
 fig.suptitle("FinQA: gold evidence in 1,147 held-out questions",
              x=.06,y=1.03,ha="left",fontsize=15,fontweight="bold",color=navy)
 fig.tight_layout(pad=2)
 for ext in ["pdf","png"]:fig.savefig(a.out/f"proposed_figure_9_26_finqa_evidence.{ext}",
                                   dpi=250,bbox_inches="tight")
 plt.close(fig)
 cmap=LinearSegmentedColormap.from_list("book",["#f3f6f9","#b5d6df","#2b7289","#153954"])
 fig,ax=plt.subplots(figsize=(10.5,5.1));im=ax.imshow(matrix,cmap=cmap,aspect="auto")
 ax.set_xticks(range(3),["Table only","Table + text","Text only"])
 ax.set_yticks(range(4),["Arithmetic","Single-span","Multi-span","Count"])
 ax.set(xlabel="Gold evidence modality",ylabel="Answer type")
 for i in range(4):
  for j in range(3):
   v=int(matrix[i,j])
   ax.text(j,i,str(v),ha="center",va="center",
           color="white" if v>matrix.max()*.55 else navy,weight="bold",fontsize=12)
 ax.set_title("TAT-QA: 1,663 held-out questions by evidence type",
              loc="left",pad=22,color=navy,weight="bold",fontsize=14)
 fig.colorbar(im,ax=ax,fraction=.035,pad=.03).set_label("Official test-question count")
 fig.tight_layout(pad=1.8)
 for ext in ["pdf","png"]:fig.savefig(a.out/f"proposed_figure_9_27_tatqa_evidence_matrix.{ext}",
                                   dpi=250,bbox_inches="tight")
 print(json.dumps({"finqa_question_count":len(finqa),"finqa_modality":cat,
                  "finqa_gold_count_bins":bins,"tatqa_question_count":len(qa),
                  "tatqa_matrix":matrix.tolist()},indent=2))
if __name__=="__main__":main()
