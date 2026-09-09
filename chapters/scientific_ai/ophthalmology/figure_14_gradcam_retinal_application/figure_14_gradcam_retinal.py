"""Scientific AI Figure 14 — Grad-CAM Explainability Pipeline for Retinal Classification.

Real-data application experiment:
- RetinaMNIST (MedMNIST) fundus images
- executable CNN training on official training split
- held-out test inference
- Grad-CAM from final convolutional layer
- three panels rendered directly from the selected real test image and tensors
No synthetic image, hand-painted heatmap, or fabricated metric is used.
"""
from pathlib import Path
import json, random, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

HERE=Path(__file__).resolve().parent
SEED=1729
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device=torch.device("cpu")
flag="retinamnist"; info=INFO[flag]; DataClass=getattr(medmnist,info["python_class"])
tfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])])
train=DataClass(split="train",transform=tfm,download=True,size=28)
val=DataClass(split="val",transform=tfm,download=True,size=28)
test=DataClass(split="test",transform=tfm,download=True,size=28)
g=torch.Generator().manual_seed(SEED)
train_loader=DataLoader(train,batch_size=128,shuffle=True,generator=g)
val_loader=DataLoader(val,batch_size=256,shuffle=False)

class RetinalCNN(nn.Module):
 def __init__(self,nc=5):
  super().__init__()
  self.features=nn.Sequential(
   nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.BatchNorm2d(32),nn.MaxPool2d(2),
   nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.BatchNorm2d(64),nn.MaxPool2d(2),
   nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.BatchNorm2d(128))
  self.pool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(128,nc)
 def forward(self,x):
  z=self.features(x); return self.fc(self.pool(z).flatten(1))

model=RetinalCNN(len(info["label"])).to(device)
opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
lossfn=nn.CrossEntropyLoss()
history=[]
best=float("inf"); best_state=None
for epoch in range(1,13):
 model.train(); tl=[]
 for x,y in train_loader:
  y=y.squeeze(1).long(); opt.zero_grad(); logits=model(x); loss=lossfn(logits,y); loss.backward(); opt.step(); tl.append(loss.item())
 model.eval(); vl=[]; correct=n=0
 with torch.no_grad():
  for x,y in val_loader:
   y=y.squeeze(1).long(); logits=model(x); vl.append(lossfn(logits,y).item()); correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
 row={"epoch":epoch,"train_loss":float(np.mean(tl)),"val_loss":float(np.mean(vl)),"val_accuracy":correct/n}; history.append(row)
 if row["val_loss"]<best: best=row["val_loss"]; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
model.load_state_dict(best_state); pd.DataFrame(history).to_csv(HERE/"figure_14_training_history.csv",index=False)

# Deterministically choose first correctly classified held-out test example.
model.eval(); chosen=None
for idx in range(len(test)):
 x,y=test[idx]; yy=int(np.asarray(y).reshape(-1)[0])
 with torch.no_grad(): pred=int(model(x.unsqueeze(0)).argmax(1))
 if pred==yy: chosen=(idx,x,yy,pred); break
idx,x,true_y,pred_y=chosen

# Grad-CAM: hooks on final convolutional layer.
target_layer=model.features[8]
acts={}; grads={}
def fh(m,i,o): acts["v"]=o
def bh(m,gi,go): grads["v"]=go[0]
h1=target_layer.register_forward_hook(fh); h2=target_layer.register_full_backward_hook(bh)
model.zero_grad(); logits=model(x.unsqueeze(0)); score=logits[0,pred_y]; score.backward()
A=acts["v"][0].detach(); G=grads["v"][0].detach(); alpha=G.mean(dim=(1,2))
cam=torch.relu((alpha[:,None,None]*A).sum(0)); cam=cam/(cam.max()+1e-12)
cam=torch.nn.functional.interpolate(cam[None,None],size=(28,28),mode="bilinear",align_corners=False)[0,0].numpy()
h1.remove(); h2.remove()
img=((x.permute(1,2,0).numpy()*.5)+.5).clip(0,1)
np.save(HERE/"figure_14_source_image.npy",img); np.save(HERE/"figure_14_gradcam.npy",cam)
pd.DataFrame({"class_index":range(len(info["label"])),"logit":logits.detach().numpy()[0]}).to_csv(HERE/"figure_14_logits.csv",index=False)

# Scientific 3-panel rendering: no title/caption/prose in artwork.
fig,axs=plt.subplots(1,3,figsize=(10.5,3.7))
axs[0].imshow(img); axs[0].set_axis_off()
im=axs[1].imshow(cam,cmap="magma",vmin=0,vmax=1); axs[1].set_axis_off()
axs[2].imshow(img); axs[2].imshow(cam,cmap="magma",alpha=.48,vmin=0,vmax=1); axs[2].set_axis_off()
for i,ax in enumerate(axs): ax.text(.5,-.08,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=11)
fig.tight_layout(w_pad=1.3)
fig.savefig(HERE/"figure_14_gradcam_retinal_classification.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_14_gradcam_retinal_classification.png",dpi=300,bbox_inches="tight"); plt.close(fig)

prov={"dataset":"RetinaMNIST / MedMNIST","dataset_version":"MedMNIST v2","official_splits":True,
 "train_examples":len(train),"validation_examples":len(val),"test_examples":len(test),
 "seed":SEED,"model":"RetinalCNN trained by included executable code","epochs":12,
 "gradcam_target_layer":"features[8], final Conv2d(64,128,3,padding=1)",
 "selected_test_index":idx,"true_class_index":true_y,"predicted_class_index":pred_y,
 "selection_rule":"first correctly classified example in official test split",
 "source_image_saved":"figure_14_source_image.npy","gradcam_tensor_saved":"figure_14_gradcam.npy",
 "classification":"Application","section":"Artificial Intelligence for Ophthalmology and Retinal Imaging",
 "subsection":"Application: Data Augmentation and Gradient-Based Explainability for Retinal Classifiers",
 "note":"Grad-CAM is computed by backpropagating the predicted-class score to the final convolutional layer; heatmap is not manually drawn."}
(HERE/"figure_14_provenance.json").write_text(json.dumps(prov,indent=2)+"\n")
