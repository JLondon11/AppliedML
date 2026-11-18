"""Decoder-only transformer fine-tuning on WikiText-103 with AdamW and Adafactor."""
import argparse, json, math, time

def main():
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Adafactor
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='distilgpt2'); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--batch',type=int,default=4); ap.add_argument('--out',default='ce03_results.json'); a=ap.parse_args()
    torch.manual_seed(7); device='cuda' if torch.cuda.is_available() else 'cpu'; ds=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1'); tok=AutoTokenizer.from_pretrained(a.model); tok.pad_token=tok.eos_token
    train_raw=ds['train'].filter(lambda x:len(x['text'].strip())>0)
    valid_raw=ds['validation'].filter(lambda x:len(x['text'].strip())>0)
    def enc(batch): return tok(batch['text'],truncation=True,max_length=256)
    train=train_raw.select(range(min(50000,len(train_raw)))).map(enc,batched=True,remove_columns=['text']); valid=valid_raw.select(range(min(3000,len(valid_raw)))).map(enc,batched=True,remove_columns=['text']); coll=DataCollatorForLanguageModeling(tok,mlm=False)
    tl=DataLoader(train,batch_size=a.batch,shuffle=True,collate_fn=coll); vl=DataLoader(valid,batch_size=a.batch,collate_fn=coll)
    results=[]
    for name in ['adamw','adafactor']:
        torch.manual_seed(7); model=AutoModelForCausalLM.from_pretrained(a.model).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01) if name=='adamw' else Adafactor(model.parameters(),lr=5e-4,relative_step=False,scale_parameter=False,weight_decay=.01)
        it=iter(tl); t0=time.perf_counter()
        for _ in range(a.steps):
            try: batch=next(it)
            except StopIteration: it=iter(tl); batch=next(it)
            batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad(); loss=model(**batch).loss; loss.backward(); opt.step()
        model.eval(); losses=[]
        with torch.no_grad():
            for j,batch in enumerate(vl):
                if j>=200: break
                batch={k:v.to(device) for k,v in batch.items()}; losses.append(float(model(**batch).loss))
        state_bytes=sum(v.numel()*v.element_size() for st in opt.state.values() for v in st.values() if torch.is_tensor(v)); results.append({'optimizer':name,'perplexity':math.exp(sum(losses)/len(losses)),'optimizer_state_bytes':state_bytes,'runtime_s':time.perf_counter()-t0})
        del model,opt
        if device=='cuda': torch.cuda.empty_cache()
    open(a.out,'w').write(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
