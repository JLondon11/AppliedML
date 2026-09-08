"""Figure 9: Training and Validation Dynamics for BERT Fine-Tuning.
Real IMDb fine-tuning experiment. All plotted losses are emitted by Trainer.
"""
from pathlib import Path
import json, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from huggingface_hub import HfApi
HERE=Path(__file__).resolve().parent
MODEL="distilbert/distilbert-base-uncased"; SEED=1729; NTRAIN=4000; NVAL=2000
torch.manual_seed(SEED); np.random.seed(SEED)
ds=load_dataset("stanfordnlp/imdb")
train=ds["train"].shuffle(seed=SEED).select(range(NTRAIN))
val=ds["test"].shuffle(seed=SEED).select(range(NVAL))
tok=AutoTokenizer.from_pretrained(MODEL)
def encode(b): return tok(b["text"],truncation=True,max_length=256)
train=train.map(encode,batched=True); val=val.map(encode,batched=True)
collator=DataCollatorWithPadding(tokenizer=tok)
model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2)
args=TrainingArguments(output_dir=str(HERE/"checkpoints"),num_train_epochs=3,
 per_device_train_batch_size=16,per_device_eval_batch_size=32,learning_rate=2e-5,
 weight_decay=.01,logging_strategy="steps",logging_steps=50,
 eval_strategy="steps",eval_steps=125,save_strategy="steps",save_steps=125,
 save_total_limit=1,load_best_model_at_end=True,metric_for_best_model="eval_loss",
 greater_is_better=False,report_to=[],seed=SEED)
trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=val,data_collator=collator)
trainer.train()
logs=pd.DataFrame(trainer.state.log_history)
logs.to_csv(HERE/"figure_09_trainer_log_history.csv",index=False)
tr=logs[logs["loss"].notna()][["step","epoch","loss"]].rename(columns={"loss":"training_loss"})
ev=logs[logs["eval_loss"].notna()][["step","epoch","eval_loss"]].rename(columns={"eval_loss":"validation_loss"})
tr.to_csv(HERE/"figure_09_training_loss.csv",index=False); ev.to_csv(HERE/"figure_09_validation_loss.csv",index=False)
best_i=ev["validation_loss"].astype(float).idxmin(); best=ev.loc[best_i]
meta={"dataset":"stanfordnlp/imdb","train_examples":NTRAIN,"validation_examples":NVAL,
 "sampling_seed":SEED,"model_id":MODEL,"model_revision":HfApi().model_info(MODEL).sha,
 "epochs":3,"learning_rate":2e-5,"train_batch_size":16,"eval_batch_size":32,
 "max_length":256,"logging_steps":50,"eval_steps":125,
 "best_validation_step":int(best["step"]),"best_validation_epoch":float(best["epoch"]),
 "best_validation_loss":float(best["validation_loss"]),"best_checkpoint":trainer.state.best_model_checkpoint,
 "torch_version":torch.__version__}
(HERE/"figure_09_experiment_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")
fig,ax=plt.subplots(figsize=(7.2,4.8))
ax.plot(tr["step"],tr["training_loss"],lw=1.35,label="Training")
ax.plot(ev["step"],ev["validation_loss"],marker="o",ms=4,lw=1.35,label="Validation")
ax.axvline(int(best["step"]),ls="--",lw=.9)
ax.scatter([int(best["step"])],[float(best["validation_loss"])],s=32,zorder=4)
ax.set_xlabel("Optimization step"); ax.set_ylabel("Cross-entropy loss")
ax.legend(frameon=False); ax.tick_params(direction="out"); ax.grid(False)
fig.tight_layout(); fig.savefig(HERE/"figure_09_bert_sentiment_loss_curves.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_09_bert_sentiment_loss_curves.png",dpi=300,bbox_inches="tight"); plt.close(fig)
