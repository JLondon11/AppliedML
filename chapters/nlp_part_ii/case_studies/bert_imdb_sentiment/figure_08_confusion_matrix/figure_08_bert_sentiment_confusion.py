"""Figure 8: Confusion Matrix for BERT Sentiment Classification.
Real-data experiment: fine-tune DistilBERT on IMDb train split and evaluate a
fixed held-out IMDb test subset. All cells derive from predictions, never invented.
"""
from pathlib import Path
import json, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from huggingface_hub import HfApi
HERE=Path(__file__).resolve().parent
MODEL="distilbert/distilbert-base-uncased"
SEED=1729; NTRAIN=4000; NTEST=2000
torch.manual_seed(SEED); np.random.seed(SEED)
ds=load_dataset("stanfordnlp/imdb")
train=ds["train"].shuffle(seed=SEED).select(range(NTRAIN))
test=ds["test"].shuffle(seed=SEED).select(range(NTEST))
tok=AutoTokenizer.from_pretrained(MODEL)
def enc(batch): return tok(batch["text"],truncation=True,max_length=256)
train=train.map(enc,batched=True); test=test.map(enc,batched=True)
model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2)
collator=DataCollatorWithPadding(tokenizer=tok)
args=TrainingArguments(output_dir=str(HERE/"checkpoints"),num_train_epochs=2,
 per_device_train_batch_size=16,per_device_eval_batch_size=32,learning_rate=2e-5,
 weight_decay=.01,logging_strategy="no",save_strategy="no",report_to=[],seed=SEED)
trainer=Trainer(model=model,args=args,train_dataset=train,data_collator=collator)
trainer.train()
pred=trainer.predict(test)
y=np.asarray(test["label"]); yhat=pred.predictions.argmax(1)
cm=confusion_matrix(y,yhat,labels=[0,1])
pd.DataFrame({"example_id":range(NTEST),"true_label":y,"predicted_label":yhat,
 "negative_probability":torch.softmax(torch.tensor(pred.predictions),1).numpy()[:,0],
 "positive_probability":torch.softmax(torch.tensor(pred.predictions),1).numpy()[:,1]}).to_csv(HERE/"figure_08_predictions.csv",index=False)
pd.DataFrame(cm,index=["true_negative","true_positive"],columns=["pred_negative","pred_positive"]).to_csv(HERE/"figure_08_confusion_matrix.csv")
meta={"dataset":"stanfordnlp/imdb","train_examples":NTRAIN,"held_out_test_examples":NTEST,
 "sampling_seed":SEED,"model_id":MODEL,"model_revision":HfApi().model_info(MODEL).sha,
 "epochs":2,"learning_rate":2e-5,"train_batch_size":16,"max_length":256,
 "accuracy":float(accuracy_score(y,yhat)),"f1":float(f1_score(y,yhat)),
 "tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1]),
 "torch_version":torch.__version__}
(HERE/"figure_08_experiment_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")
fig,ax=plt.subplots(figsize=(5.4,4.8))
im=ax.imshow(cm,cmap="cividis")
ax.set_xticks([0,1],["Negative","Positive"]); ax.set_yticks([0,1],["Negative","Positive"])
ax.set_xlabel("Predicted sentiment"); ax.set_ylabel("True sentiment")
for i in range(2):
 for j in range(2):
  ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",fontsize=13,
          color="white" if cm[i,j]>.55*cm.max() else "black")
fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label="Number of reviews")
fig.tight_layout(); fig.savefig(HERE/"figure_08_bert_sentiment_confusion.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_08_bert_sentiment_confusion.png",dpi=300,bbox_inches="tight"); plt.close(fig)
