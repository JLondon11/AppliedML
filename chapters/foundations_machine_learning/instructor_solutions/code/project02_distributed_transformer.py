"""Distributed-transformer reference: AMP, accumulation, optional FSDP, checkpoint recovery, throughput."""
import argparse, os, time, torch
from torch import nn

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--accum',type=int,default=4); ap.add_argument('--checkpoint',default='project02.pt'); ap.add_argument('--fsdp',action='store_true'); a=ap.parse_args()
    distributed=int(os.environ.get('WORLD_SIZE','1'))>1
    if distributed and not torch.distributed.is_initialized(): torch.distributed.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    local_rank=int(os.environ.get('LOCAL_RANK','0')); device=torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.set_device(device)
    class LM(nn.Module):
        def __init__(self): super().__init__(); self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,batch_first=True),4); self.head=nn.Linear(256,32000)
        def forward(self,x): return self.head(self.enc(x))
    model=LM().to(device)
    if a.fsdp and distributed:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model=FSDP(model)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4); start=0
    if os.path.exists(a.checkpoint):
        ck=torch.load(a.checkpoint,map_location=device); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); start=ck['step']; torch.set_rng_state(ck['rng'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda'); t0=time.perf_counter(); tokens=0
    for step in range(start,a.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            x=torch.randn(8,128,256,device=device); y=torch.randint(0,32000,(8,128),device=device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                loss=nn.functional.cross_entropy(model(x).flatten(0,1),y.flatten())/a.accum
            scaler.scale(loss).backward(); tokens+=y.numel()
        scaler.step(opt); scaler.update()
        if (step+1)%25==0 and (not distributed or torch.distributed.get_rank()==0): torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step+1,'rng':torch.get_rng_state()},a.checkpoint)
    elapsed=time.perf_counter()-t0
    if not distributed or torch.distributed.get_rank()==0: print({'tokens_per_s':tokens/elapsed,'steps':a.steps,'world_size':int(os.environ.get('WORLD_SIZE','1'))})
if __name__=='__main__': main()
