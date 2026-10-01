"""Listing 1.3 — CNN training loop instrumented for optimizer memory and accuracy.

Uses sklearn digits to stay small and CPU-runnable.
"""
from __future__ import annotations
import random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED=42

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,8,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.head=nn.Linear(16,10)
    def forward(self,x):
        return self.head(self.net(x).flatten(1))

def top1_accuracy(model,loader,device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            pred=model(xb).argmax(1)
            correct+=int((pred==yb).sum()); total+=len(yb)
    return correct/total

def optimizer_state_bytes(optimizer):
    total=0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total+=value.numel()*value.element_size()
    return total

def build_loaders():
    X,y=load_digits(return_X_y=True)
    X=(X/16.0).astype("float32").reshape(-1,1,8,8)
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    return (
        DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr,dtype=torch.long)),64,shuffle=True),
        DataLoader(TensorDataset(torch.tensor(Xv),torch.tensor(yv,dtype=torch.long)),256)
    )

def train_once(name="adamw",epochs=3):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device=torch.device("cpu")
    train_loader,val_loader=build_loaders()
    model=TinyCNN().to(device)
    if name=="sgd":
        opt=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
    elif name=="adamw":
        opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4)
    else:
        raise ValueError("name must be 'sgd' or 'adamw'")
    loss_fn=nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb,yb in train_loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
    return {
        "optimizer":name,
        "validation_accuracy":top1_accuracy(model,val_loader,device),
        "optimizer_state_bytes":optimizer_state_bytes(opt),
    }

if __name__=="__main__":
    for name in ("sgd","adamw"):
        print(train_once(name))
