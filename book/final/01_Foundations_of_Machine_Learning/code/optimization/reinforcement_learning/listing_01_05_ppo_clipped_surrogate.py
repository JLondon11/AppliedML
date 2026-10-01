"""Listing 1.5 — Core PPO clipped surrogate objective computation."""
from __future__ import annotations
import torch

def ppo_clipped_loss(new_log_probs,old_log_probs,advantages,clip_epsilon=0.2):
    ratio=torch.exp(new_log_probs-old_log_probs)
    unclipped=ratio*advantages
    clipped=torch.clamp(ratio,1-clip_epsilon,1+clip_epsilon)*advantages
    policy_loss=-torch.min(unclipped,clipped).mean()
    approx_kl=(old_log_probs-new_log_probs).mean()
    clip_fraction=((ratio-1.0).abs()>clip_epsilon).float().mean()
    return policy_loss,approx_kl,clip_fraction

if __name__=="__main__":
    torch.manual_seed(42)
    old=torch.randn(32)*.2
    new=old+torch.randn(32)*.05
    adv=torch.randn(32)
    loss,kl,cf=ppo_clipped_loss(new,old,adv)
    print({"loss":float(loss),"approx_kl":float(kl),"clip_fraction":float(cf)})
