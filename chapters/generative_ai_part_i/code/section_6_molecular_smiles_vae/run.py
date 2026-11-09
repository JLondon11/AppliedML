"""NCI character-level SMILES variational autoencoder.

Uses actual RDKit Data/NCI/first_5K.smi molecular structures.
Run with RDKit, PyTorch, numpy, scikit-learn, matplotlib:
python run.py --epochs 6 --limit 600
"""
import argparse, csv, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDConfig, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.error")
SEED=20261109

class MolVAE(nn.Module):
 def __init__(self,vocab,z=32,h=128,embed=32):
  super().__init__()
  self.emb=nn.Embedding(vocab,embed,padding_idx=0)
  self.encoder=nn.GRU(embed,h,batch_first=True,bidirectional=True)
  self.mu=nn.Linear(2*h,z); self.logvar=nn.Linear(2*h,z)
  self.init=nn.Linear(z,h)
  self.decoder=nn.GRU(embed+z,h,batch_first=True)
  self.out=nn.Linear(h,vocab)
 def encode(self,seq):
  _,hidden=self.encoder(self.emb(seq))
  state=torch.cat((hidden[0],hidden[1]),-1)
  return self.mu(state),self.logvar(state).clamp(-8,8)
 def forward(self,seq):
  mu,lv=self.encode(seq)
  z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
  emb=self.emb(seq[:,:-1])
  inp=torch.cat((emb,z[:,None].expand(-1,emb.size(1),-1)),2)
  h,_=self.decoder(inp,self.init(z)[None])
  return self.out(h),mu,lv

@torch.no_grad()
def decode(model,z,stoi,itos,maxlen,temperature=1.0,greedy=False):
 n=len(z);t=torch.full((n,),stoi["^"],dtype=torch.long)
 hidden=model.init(z)[None];strings=[""]*n;finished=[False]*n
 for _ in range(maxlen):
  embedding=model.emb(t)[:,None]
  inp=torch.cat((embedding,z[:,None,:]),2)
  out,hidden=model.decoder(inp,hidden)
  logits=model.out(out[:,0])
  t=logits.argmax(-1) if greedy else torch.multinomial(torch.softmax(logits/temperature,1),1)[:,0]
  for j,index in enumerate(t.tolist()):
   if finished[j]:continue
   if index==stoi["$"]:finished[j]=True
   elif index>3:strings[j]+=itos[index]
  if all(finished):break
 return strings

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--epochs",type=int,default=6)
 p.add_argument("--limit",type=int,default=600)
 p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"results")
 args=p.parse_args()
 out=args.output;fig=out/"figures";fig.mkdir(parents=True,exist_ok=True)
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
 path=Path(RDConfig.RDDataDir)/"NCI"/"first_5K.smi"
 sha=hashlib.sha256(path.read_bytes()).hexdigest()
 raw=path.read_text().splitlines()
 smiles=[];identifiers=[];seen=set();invalid=too_long=duplicates=0
 for line in raw:
  toks=line.strip().split()
  if len(toks)<2:continue
  mol=Chem.MolFromSmiles(toks[0])
  if mol is None:invalid+=1;continue
  s=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
  if len(s)>75:too_long+=1;continue
  if s in seen:duplicates+=1;continue
  seen.add(s);smiles.append(s);identifiers.append(toks[1])
  if len(smiles)>=args.limit:break
 assert len(smiles)>=500
 idx=np.arange(len(smiles))
 train,test=train_test_split(idx,test_size=.2,random_state=SEED)
 train,valid=train_test_split(train,test_size=.25,random_state=SEED)
 chars=sorted(set("".join(smiles)))
 itos=["<pad>","^","$","<?>"]+chars
 stoi={c:i for i,c in enumerate(itos)}
 maxlen=max(map(len,smiles))+2
 def encode(s):
  tokens=[stoi["^"]]+[stoi[c] for c in s]+[stoi["$"]]
  return tokens+[0]*(maxlen-len(tokens))
 seq=torch.tensor([encode(s) for s in smiles],dtype=torch.long)
 split={int(i):name for name,ids in (("train",train),("validation",valid),("test",test)) for i in ids}
 with (out/"source_manifest.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["source_id","canonical_smiles","split"])
  w.writerows((identifiers[i],smiles[i],split[i]) for i in idx)
 model=MolVAE(len(itos));optimizer=torch.optim.Adam(model.parameters(),lr=.003)
 trace=[];best=float("inf");chosen=0;state=None
 for epoch in range(1,args.epochs+1):
  model.train();order=torch.randperm(len(train));cumulative=0
  beta=min(.12,epoch*.015)
  for batch in order.split(64):
   x=seq[train[batch]]
   logits,mu,lv=model(x)
   rec=nn.functional.cross_entropy(logits.reshape(-1,len(itos)),x[:,1:].reshape(-1),ignore_index=0)
   kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(-1)).mean()
   loss=rec+beta*kl/32
   optimizer.zero_grad();loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),5)
   optimizer.step();cumulative+=float(loss.detach())*len(batch)
  model.eval();losses=[]
  with torch.no_grad():
   for batch in torch.tensor(valid).split(128):
    x=seq[batch];logits,_,_=model(x)
    v=nn.functional.cross_entropy(logits.reshape(-1,len(itos)),x[:,1:].reshape(-1),ignore_index=0)
    losses.append((float(v),len(batch)))
  val=sum(v*n for v,n in losses)/sum(n for v,n in losses)
  trace.append((epoch,cumulative/len(train),val,beta))
  if val<best:
   best=val;chosen=epoch
   state={k:v.detach().clone() for k,v in model.state_dict().items()}
  print(epoch,trace[-1],flush=True)
 model.load_state_dict(state);model.eval()
 generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=512)
 test_records=[]
 for batch in np.array_split(np.sort(test),max(1,(len(test)+63)//64)):
  with torch.no_grad():
   mu,_=model.encode(seq[batch])
   reconstructed=decode(model,mu,stoi,itos,maxlen,greedy=True)
  for index,s in zip(batch,reconstructed):
   mol=Chem.MolFromSmiles(s) if s else None
   canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False) if mol else ""
   true=Chem.MolFromSmiles(smiles[index])
   tan=float(DataStructs.TanimotoSimilarity(generator.GetFingerprint(true),generator.GetFingerprint(mol))) if mol else 0.
   test_records.append((identifiers[index],smiles[index],s,canonical,int(bool(mol)),int(canonical==smiles[index]),tan))
 with (out/"heldout_reconstructions.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["source_id","reference_smiles","decoded_smiles","canonical_decoded","valid","exact_reconstruction","morgan_tanimoto"]);w.writerows(test_records)
 torch.manual_seed(SEED+999);samples=[]
 with torch.no_grad():
  for _ in range(8):
   samples.extend(decode(model,torch.randn(128,32),stoi,itos,maxlen,temperature=.85))
 train_set={smiles[i] for i in train}
 known=set(smiles)
 prior=[]
 for s in samples:
  mol=Chem.MolFromSmiles(s) if s else None
  canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False) if mol else ""
  prior.append((s,canonical,int(bool(mol)),int(bool(canonical) and canonical not in train_set),int(bool(canonical) and canonical not in known)))
 with (out/"prior_samples.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["raw_sample","canonical_smiles","valid","novel_to_training","novel_to_full_data"]);w.writerows(prior)
 usable=[r[1] for r in prior if r[2]]
 novelty=[s for s in usable if s not in train_set]
 m={"dataset":"RDKit bundled NCI first_5K.smi","source_sha256":sha,"source_records":len(raw),
  "filtered_unique_molecules":len(smiles),"excluded_invalid":invalid,"excluded_length":too_long,"excluded_duplicate":duplicates,
  "n_train":len(train),"n_validation":len(valid),"n_test":len(test),"seed":SEED,
  "max_sequence_len":maxlen,"latent_dim":32,"epochs":args.epochs,"checkpoint_epoch":chosen,
  "test_exact_reconstruction":float(np.mean([r[5] for r in test_records])),
  "test_validity":float(np.mean([r[4] for r in test_records])),
  "test_mean_morgan_tanimoto":float(np.mean([r[6] for r in test_records])),
  "n_prior_samples":len(prior),"prior_validity":float(np.mean([r[2] for r in prior])),
  "prior_unique_valid_fraction":len(set(usable))/max(1,len(usable)),
  "prior_novel_to_train_fraction_among_valid":len(novelty)/max(1,len(usable)),
  "prior_novel_unique_count":len(set(novelty)),
  "prior_novel_to_full_data_fraction_among_valid":float(np.mean([r[4] for r in prior if r[2]])) if usable else 0.,
  "limitations":["small convenience subset","random structure split, not scaffold split","low prior validity","no property conditioning or biological endpoint"]}
 (out/"metrics.json").write_text(json.dumps(m,indent=2))
 with (out/"training.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["epoch","training_objective","validation_crossentropy","kl_weight"]);w.writerows(trace)
 torch.save({"weights":state,"stoi":stoi,"itos":itos,"metrics":m},out/"smiles_vae.pt")
 fig1,ax=plt.subplots(figsize=(7.5,3.5))
 ax.plot([t[0] for t in trace],[t[1] for t in trace],label="Training objective")
 ax.plot([t[0] for t in trace],[t[2] for t in trace],label="Validation token CE")
 ax.set(xlabel="Epoch",ylabel="Loss (different objectives)")
 ax.legend(frameon=False);fig1.tight_layout()
 fig1.savefig(fig/"molecular_smiles_vae_training.png",dpi=190);plt.close(fig1)
 fig2,ax=plt.subplots(figsize=(9,3.6))
 names=["Test valid","Test exact","Prior valid","Unique among valid","Novel among valid"]
 values=[m[k] for k in ["test_validity","test_exact_reconstruction","prior_validity","prior_unique_valid_fraction","prior_novel_to_train_fraction_among_valid"]]
 ax.bar(range(5),values);ax.set(xticks=range(5),xticklabels=names,ylim=(0,1.05),ylabel="Proportion (different denominators)")
 ax.grid(axis="y",alpha=.2);fig2.tight_layout()
 fig2.savefig(fig/"molecular_smiles_generation_results.png",dpi=190);plt.close(fig2)
 print(json.dumps(m,indent=2))

if __name__=="__main__":main()
