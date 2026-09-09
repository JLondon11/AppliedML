"""Shared real-data experiment for NLP Part II Figures 8 and 9.

Resource-conscious reproduction:
- model: google/bert_uncased_L-2_H-128_A-2 (BERT-Tiny architecture)
- data: Stanford IMDb
- train: 2,000 reviews from official train split
- validation: 500 disjoint reviews from official train split
- test: 1,000 reviews from official test split, never used for checkpoint choice
- 3 epochs, CPU-compatible
- Figure 9 uses measured train/validation loss
- Figure 8 uses predictions from the best validation-loss checkpoint

No numerical result is hard-coded.
"""
from pathlib import Path
import json, shutil, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from huggingface_hub import HfApi

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
F8=ROOT/"figure_08_confusion_matrix"; F9=ROOT/"figure_09_loss_curves"
F8.mkdir(exist_ok=True); F9.mkdir(exist_ok=True)
MODEL="google/bert_uncased_L-2_H-128_A-2"; SEED=1729
NTRAIN=2000; NVAL=500; NTEST=1000
torch.manual_seed(SEED); np.random.seed(SEED)

ds=load_dataset("stanfordnlp/imdb")
pool=ds["train"].shuffle(seed=SEED)
train=pool.select(range(NTRAIN))
val=pool.select(range(NTRAIN,NTRAIN+NVAL))
test=ds["test"].shuffle(seed=SEED).select(range(NTEST))
tok=AutoTokenizer.from_pretrained(MODEL)
def encode(b): return tok(b["text"],truncation=True,max_length=192)
train=train.map(encode,batched=True); val=val.map(encode,batched=True); test=test.map(encode,batched=True)
collator=DataCollatorWithPadding(tokenizer=tok)
model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2)
args=TrainingArguments(
 output_dir=str(HERE/"checkpoints"),num_train_epochs=3,
 per_device_train_batch_size=32,per_device_eval_batch_size=64,
 learning_rate=3e-5,weight_decay=.01,
 logging_strategy="steps",logging_steps=10,
 eval_strategy="epoch",save_strategy="epoch",save_total_limit=1,
 load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,
 report_to=[],seed=SEED,dataloader_num_workers=0)
trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=val,data_collator=collator)
trainer.train()

logs=pd.DataFrame(trainer.state.log_history)
logs.to_csv(HERE/"trainer_log_history.csv",index=False)
tr=logs[logs["loss"].notna()][["step","epoch","loss"]].rename(columns={"loss":"training_loss"})
ev=logs[logs["eval_loss"].notna()][["step","epoch","eval_loss"]].rename(columns={"eval_loss":"validation_loss"})
tr.to_csv(F9/"figure_09_training_loss.csv",index=False); ev.to_csv(F9/"figure_09_validation_loss.csv",index=False)
best=ev.loc[ev["validation_loss"].astype(float).idxmin()]

# Figure 9: actual learning dynamics.
fig,ax=plt.subplots(figsize=(7.2,4.8))
ax.plot(tr["step"],tr["training_loss"],lw=1.35,label="Training")
ax.plot(ev["step"],ev["validation_loss"],marker="o",ms=4,lw=1.35,label="Validation")
ax.axvline(int(best["step"]),ls="--",lw=.9)
ax.scatter([int(best["step"])],[float(best["validation_loss"])],s=32,zorder=4)
ax.set_xlabel("Optimization step"); ax.set_ylabel("Cross-entropy loss")
ax.legend(frameon=False); ax.tick_params(direction="out"); ax.grid(False)
fig.tight_layout(); fig.savefig(F9/"figure_09_bert_sentiment_loss_curves.svg",bbox_inches="tight")
fig.savefig(F9/"figure_09_bert_sentiment_loss_curves.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# Figure 8: untouched official-test evaluation using selected checkpoint.
pred=trainer.predict(test); logits=pred.predictions
prob=torch.softmax(torch.tensor(logits),1).numpy()
y=np.asarray(test["label"]); yhat=logits.argmax(1)
cm=confusion_matrix(y,yhat,labels=[0,1])
pd.DataFrame({"example_id":range(NTEST),"true_label":y,"predicted_label":yhat,
 "negative_probability":prob[:,0],"positive_probability":prob[:,1]}).to_csv(F8/"figure_08_predictions.csv",index=False)
pd.DataFrame(cm,index=["true_negative","true_positive"],columns=["pred_negative","pred_positive"]).to_csv(F8/"figure_08_confusion_matrix.csv")

fig,ax=plt.subplots(figsize=(5.4,4.8)); im=ax.imshow(cm,cmap="cividis")
ax.set_xticks([0,1],["Negative","Positive"]); ax.set_yticks([0,1],["Negative","Positive"])
ax.set_xlabel("Predicted sentiment"); ax.set_ylabel("True sentiment")
for i in range(2):
 for j in range(2):
  ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",fontsize=13,
          color="white" if cm[i,j]>.55*cm.max() else "black")
fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label="Number of reviews")
fig.tight_layout(); fig.savefig(F8/"figure_08_bert_sentiment_confusion.svg",bbox_inches="tight")
fig.savefig(F8/"figure_08_bert_sentiment_confusion.png",dpi=300,bbox_inches="tight"); plt.close(fig)

meta={"dataset":"stanfordnlp/imdb","model_id":MODEL,"model_revision":HfApi().model_info(MODEL).sha,
 "seed":SEED,"train_examples":NTRAIN,"validation_examples":NVAL,"test_examples":NTEST,
 "split_protocol":"train/validation are disjoint samples from official train split; test is sampled only from official test split",
 "epochs":3,"learning_rate":3e-5,"train_batch_size":32,"eval_batch_size":64,"max_length":192,
 "best_validation_step":int(best["step"]),"best_validation_epoch":float(best["epoch"]),
 "best_validation_loss":float(best["validation_loss"]),
 "test_accuracy":float(accuracy_score(y,yhat)),"test_f1":float(f1_score(y,yhat)),
 "tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1]),
 "torch_version":torch.__version__}
(HERE/"experiment_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")
(F8/"figure_08_experiment_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")
(F9/"figure_09_experiment_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")
shutil.rmtree(HERE/"checkpoints",ignore_errors=True)
