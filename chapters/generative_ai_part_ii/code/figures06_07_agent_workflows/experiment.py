"""Real-data tool-orchestration experiment for candidate Chapter 9 Figures 6–7.

Three deterministic classifier-tool workflows on the 1797 real sklearn digits.
NOT autonomous LLM agents. Each decision and verification tool call is logged.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS=["#294C60","#A96543","#5A7D7C"]
NAMES=["planner-executor","supervisor-worker","peer-vote"]

def evaluate(p, threshold):
    events=[];results={}
    n=len(p["linear"])
    for mode,primary,backup in [(NAMES[0],"linear","forest"),(NAMES[1],"forest","neighbors")]:
        first,second=p[primary],p[backup]
        conf=first.max(1);consult=conf<threshold
        pred=np.where(consult,second.argmax(1),first.argmax(1))
        results[mode]={"prediction":pred,"consulted":consult,"confidence":conf}
        for j in range(n):
            ev=[("primary_tool",primary,int(first[j].argmax()),float(conf[j])),
                ("verify_confidence","threshold_check",int(first[j].argmax()),float(conf[j]))]
            if consult[j]:ev.append(("fallback_tool",backup,int(second[j].argmax()),float(second[j].max())))
            ev.append(("final_answer",backup if consult[j] else primary,int(pred[j]),
                       float(second[j].max() if consult[j] else conf[j])))
            for step,(event,tool,prediction,confidence) in enumerate(ev,1):
                events.append({"workflow":mode,"test_position":j,"step":step,"event":event,
                               "tool":tool,"prediction":prediction,"confidence":confidence})
    tools=["linear","forest","neighbors"]
    allpred=np.stack([p[k].argmax(1) for k in tools],axis=1)
    votes=[];agreements=[]
    for j in range(n):
        c=np.bincount(allpred[j],minlength=10);v=int(p["forest"][j].argmax()) if np.sum(c==c.max())>1 else int(c.argmax())
        votes.append(v);agreements.append(float(c.max()/3))
        for k,tool in enumerate(tools):
            events.append({"workflow":NAMES[2],"test_position":j,"step":k+1,"event":"peer_tool",
                           "tool":tool,"prediction":int(allpred[j,k]),"confidence":float(p[tool][j].max())})
        events.extend([{"workflow":NAMES[2],"test_position":j,"step":4,"event":"verify_vote",
                        "tool":"majority_with_forest_tiebreak","prediction":v,"confidence":float(c.max()/3)},
                       {"workflow":NAMES[2],"test_position":j,"step":5,"event":"final_answer",
                        "tool":"majority_vote","prediction":v,"confidence":float(c.max()/3)}])
    results[NAMES[2]]={"prediction":np.asarray(votes),"consulted":np.ones(n,dtype=bool),
                      "confidence":np.asarray(agreements)}
    return results,events

def main(out,seed=41):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    digits=load_digits()
    X=digits.data.astype("float64")/16.;y=digits.target
    ids=np.arange(len(y))
    trval,test=train_test_split(ids,test_size=270,random_state=seed,stratify=y)
    train,val=train_test_split(trval,test_size=270,random_state=seed,stratify=y[trval])
    assert (len(train),len(val),len(test))==(1257,270,270)
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    models={"linear":make_pipeline(StandardScaler(),LogisticRegression(max_iter=700,random_state=seed)),
            "forest":RandomForestClassifier(n_estimators=180,min_samples_leaf=2,n_jobs=1,random_state=seed),
            "neighbors":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7))}
    for model in models.values():model.fit(X[train],y[train])
    pv={k:m.predict_proba(X[val]) for k,m in models.items()}
    pt={k:m.predict_proba(X[test]) for k,m in models.items()}
    trials=[]
    for t in (.5,.6,.7,.8,.9):
        r,_=evaluate(pv,t)
        trials.append((float(np.mean([accuracy_score(y[val],r[w]["prediction"]) for w in NAMES[:2]])),float(t)))
    _,t=max(trials,key=lambda a:(a[0],-a[1]))
    results,events=evaluate(pt,t)
    report=[]
    for name in NAMES:
        r=results[name]
        report.append({"workflow":name,"test_accuracy":float(accuracy_score(y[test],r["prediction"])),
                       "correct":int(np.sum(r["prediction"]==y[test])),
                       "extra_tool_consultation_fraction":float(r["consulted"].mean()),
                       "mean_decision_confidence":float(r["confidence"].mean()),
                       "recorded_events":sum(e["workflow"]==name for e in events),
                       "confusion_matrix":confusion_matrix(y[test],r["prediction"],labels=np.arange(10)).tolist()})
    for event in events:event["source_index"]=int(test[event["test_position"]])
    with (out/"observed_tool_events.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["workflow","test_position","source_index","step","event",
                                      "tool","prediction","confidence"])
        w.writeheader();w.writerows(events)
    with (out/"test_predictions.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["source_index","actual",*NAMES])
        for j,index in enumerate(test):w.writerow([int(index),int(y[index])]+[int(results[k]["prediction"][j]) for k in NAMES])
    fig,ax=plt.subplots(1,3,figsize=(12,3.6))
    labels=["Planner–executor","Supervisor–worker","Peer vote"]
    for axis,key,label in zip(ax,["test_accuracy","extra_tool_consultation_fraction","recorded_events"],
                              ["Independent test accuracy","Extra model consultation rate","Mean logged events per image"]):
        values=[r[key]/270 if key=="recorded_events" else r[key] for r in report]
        axis.bar(np.arange(3),values,color=COLORS,width=.58)
        axis.set_xticks(range(3),labels,rotation=18)
        axis.set_ylabel(label)
        if key!="recorded_events":axis.set_ylim(0,1.05)
    fig.suptitle("Actual multi-model orchestration on 270 real held-out handwritten digits")
    fig.tight_layout();fig.savefig(out/"Figure_006_Three_multi_agent_architectural_patterns.png",dpi=200);plt.close(fig)
    recovery=np.flatnonzero((pt["linear"].argmax(1)!=y[test])&
                             (pt["forest"].argmax(1)==y[test])&
                             (pt["linear"].max(1)<t))
    pos=int(recovery[0]) if len(recovery) else int(np.argmin(results[NAMES[0]]["confidence"]))
    trace=[e for e in events if e["workflow"]==NAMES[0] and e["test_position"]==pos]
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),gridspec_kw={"width_ratios":[1,2.7]})
    ax[0].imshow(X[test[pos]].reshape(8,8),cmap="gray",vmin=0,vmax=1)
    ax[0].set_title(f"Real test digit, label {y[test[pos]]}");ax[0].axis("off")
    for j,e in enumerate(trace):
        yval=e["confidence"]
        ax[1].scatter(j,yval,color=COLORS[0],s=45)
        ax[1].annotate(f'{e["event"].replace("_"," ")}\n{e["tool"]} → {e["prediction"]}',
                       (j,yval),xytext=(0,15 if j%2==0 else -46),
                       textcoords="offset points",ha="center",fontsize=7)
    ax[1].plot(range(len(trace)),[e["confidence"] for e in trace],color=COLORS[0],lw=1)
    ax[1].axhline(t,color=COLORS[1],ls="--",label=f"Validation threshold = {t:.2f}")
    ax[1].set(xlim=(-.45,len(trace)-.55),ylim=(-.13,1.12),
              xticks=range(len(trace)),xticklabels=[str(e["step"]) for e in trace],
              xlabel="Observed event step",ylabel="Recorded tool confidence")
    ax[1].legend(loc="upper right",fontsize=8)
    fig.suptitle("Real-image tool selection, verification and fallback execution trace")
    fig.tight_layout();fig.savefig(out/"Figure_007_Agent_planning_and_verification_loop.png",dpi=200);plt.close(fig)
    metrics={"dataset_name":"scikit-learn real handwritten digits",
             "dataset_source_url":"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
             "split_counts":{"train":1257,"validation":270,"test":270},"split_disjoint":True,
             "split_seed":seed,"threshold_validation_scores":trials,
             "validation_selected_threshold":t,"model_tools":list(models),
             "base_test_accuracies":{k:float(accuracy_score(y[test],v.argmax(1))) for k,v in pt.items()},
             "workflow_results":report,"actual_trace_test_source_index":int(test[pos]),
             "actual_trace_label":int(y[test[pos]]),"actual_trace_event_count":len(trace),
             "total_test_events":len(events),
             "method":"deterministic classifier-tool workflows and observed event traces on real digit images",
             "limitations":"Real executed tool orchestration, not LLM agents, autonomous planning, or multi-agent benchmark performance."}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    main(a.out)
