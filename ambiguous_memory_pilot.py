#!/usr/bin/env python3
"""Action-ambiguous Q-memory pilot: action text cannot reveal the answer."""
import argparse, csv, json, math, random, statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

VOCAB=["<pad>","noise","earlier","state_a","state_b","update","left","right","open"]
STOI={w:i for i,w in enumerate(VOCAB)}

def data(n,seed,length=24):
    rng=random.Random(seed); rows=[]
    for _ in range(n):
        truth=rng.randrange(2); stale=1-truth
        seq=["noise"]*rng.randint(1,5)+["earlier", "state_a" if stale==0 else "state_b"]
        seq += ["noise"]*rng.randint(1,5)+["update", "state_a" if truth==0 else "state_b"]
        seq += ["noise"]*rng.randint(1,5)
        ids=[STOI[x] for x in seq[:length]]+[0]*max(0,length-len(seq))
        rows.append((ids,truth))
    return torch.tensor([r[0] for r in rows]), torch.tensor([r[1] for r in rows])

class Memory(nn.Module):
    def __init__(self,codes,hidden=48):
        super().__init__(); self.emb=nn.Embedding(len(VOCAB),24,padding_idx=0); self.gru=nn.GRU(24,hidden,batch_first=True); self.to_code=nn.Linear(hidden,codes); self.head=nn.Linear(codes,2,bias=False)
    def forward(self,x,tau=1,hard=True):
        _,h=self.gru(self.emb(x)); lg=self.to_code(h[-1]); z=F.gumbel_softmax(lg,tau=tau,hard=hard); return lg,self.head(z)

def run(codes,seed,a,device):
    torch.manual_seed(seed); x,y=data(a.train,1000+seed); vx,vy=data(a.test,9000+seed); x,y,vx,vy=[z.to(device) for z in (x,y,vx,vy)]
    m=Memory(codes,a.hidden).to(device); opt=torch.optim.AdamW(m.parameters(),lr=a.lr)
    for ep in range(a.epochs):
        tau=max(.35,1.5*(1-ep/a.epochs)); lg,p=m(x,tau,True); loss=F.cross_entropy(p,y); marginal=F.softmax(lg/tau,-1).mean(0).clamp_min(1e-8); loss+=a.balance*(marginal*(marginal.log()+math.log(codes))).sum(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): lg,p=m(vx,.1,True); code=lg.argmax(1); acc=(p.argmax(1)==vy).float().mean().item(); probe={}
    for c in code.unique().tolist(): probe[c]=float(vy[code==c].float().mean())
    return {"codes":codes,"seed":seed,"accuracy":acc,"used_codes":int(code.unique().numel()),"probe":probe,"device":str(device)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--train",type=int,default=6000); p.add_argument("--test",type=int,default=2000); p.add_argument("--codes",default="1,2,4,8"); p.add_argument("--seeds",type=int,default=5); p.add_argument("--epochs",type=int,default=150); p.add_argument("--hidden",type=int,default=48); p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--balance",type=float,default=.05); p.add_argument("--out",default="ambiguous_memory_results"); a=p.parse_args(); d=torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows=[]
    for c in map(int,a.codes.split(",")):
        for s in range(a.seeds): rows.append(run(c,s,a,d)); print(rows[-1],flush=True)
    Path(a.out).mkdir(exist_ok=True); json.dump({"config":vars(a),"device":str(d),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"rows":rows},open(Path(a.out)/"results.json","w"),indent=2)
if __name__=="__main__": main()
