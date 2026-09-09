#!/usr/bin/env python3
"""Learned discrete-memory pilot for Q-memory versus reconstruction memory."""
import argparse, csv, json, math, os, random, time
import torch
from torch import nn
from torch.nn import functional as F

SPECIAL = ["<pad>", "start", "earlier", "was_open", "update", "is_open", "door_a", "door_b"]
DECOR = ["marble", "quiet", "copper", "old", "garden", "blue", "small", "round", "bright", "wooden", "dusty", "warm"]
VOCAB = SPECIAL + DECOR
STOI = {w: i for i, w in enumerate(VOCAB)}

def dataset(n, seed, length=32):
    rng = random.Random(seed); xs=[]; ys=[]; bows=[]
    for _ in range(n):
        truth = rng.randrange(2); stale = 1-truth
        # Ordered evidence makes the latest-update rule observable from history.
        seq = ["start"] + [rng.choice(DECOR) for _ in range(rng.randint(2,7))]
        seq += ["earlier", f"door_{'a' if stale==0 else 'b'}", "was_open"]
        seq += [rng.choice(DECOR) for _ in range(rng.randint(4,10))]
        seq += ["update", f"door_{'a' if truth==0 else 'b'}", "is_open"]
        seq += [rng.choice(DECOR) for _ in range(rng.randint(2,8))]
        ids=[STOI[w] for w in seq[:length]]; ids += [0]*(length-len(ids))
        bow=[0.0]*len(VOCAB)
        for t in ids:
            if t: bow[t] += 1.0
        s=max(1.0,sum(bow)); bow=[x/s for x in bow]
        xs.append(ids); ys.append([1.0,-1.0] if truth==0 else [-1.0,1.0]); bows.append(bow)
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(bows)

class Memory(nn.Module):
    def __init__(self, codes, hidden=64, length=32):
        super().__init__(); self.codes=codes
        self.length=length
        self.emb=nn.Embedding(len(VOCAB),32,padding_idx=0); self.gru=nn.GRU(32,hidden,batch_first=True)
        self.to_code=nn.Linear(hidden,codes); self.q_head=nn.Linear(codes,2,bias=False)
        self.recon_head=nn.Linear(codes,length*len(VOCAB),bias=False)
    def encode(self,x,tau=1.0,hard=True):
        _,h=self.gru(self.emb(x)); logits=self.to_code(h[-1])
        return logits, F.gumbel_softmax(logits,tau=tau,hard=hard)

    def deterministic_code(self, logits):
        return F.one_hot(logits.argmax(-1), self.codes).float()

def batches(x,y,b,batch,seed):
    g=torch.Generator().manual_seed(seed); order=torch.randperm(len(x),generator=g)
    for i in range(0,len(x),batch):
        ix=order[i:i+batch]; yield x[ix],y[ix],b[ix]

def probe_accuracy(codes, y, train_codes=None, train_y=None):
    if train_codes is None: train_codes,train_y=codes,y
    mapping={}
    for c in train_codes.unique().tolist():
        mask=train_codes==c; mapping[c]=int(train_y[mask].argmax(1).float().mean()>=.5)
    pred=torch.tensor([mapping.get(int(c),0) for c in codes]); target=y.argmax(1).cpu()
    return (pred==target).float().mean().item()

def run_one(objective,codes,seed,args,device):
    torch.manual_seed(seed); train=dataset(args.train,1000+seed); test=dataset(args.test,9000+seed)
    train_fit=train
    if objective=="q_permuted":
        g=torch.Generator().manual_seed(70000+seed)
        train_fit=(train[0],train[1][torch.randperm(len(train[1]),generator=g)],train[2])
    model=Memory(codes,args.hidden,args.length).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    start=time.time()
    for epoch in range(args.epochs):
        model.train(); tau=max(.35,1.5*(1-epoch/max(1,args.epochs)))
        for x,y,b in batches(*train_fit,args.batch,seed*100+epoch):
            x,y,b=x.to(device),y.to(device),b.to(device); code_logits,z=model.encode(x,tau,True)
            if objective in ("q","q_permuted"): loss=F.smooth_l1_loss(model.q_head(z),y)
            else:
                logits=model.recon_head(z).view(-1,args.length,len(VOCAB))
                loss=F.cross_entropy(logits.reshape(-1,len(VOCAB)),x.reshape(-1),ignore_index=0)
                # Prevent a weak reconstruction baseline from silently using less
                # than its declared code budget.
                marginal=F.softmax(code_logits/tau,-1).mean(0).clamp_min(1e-8)
                loss=loss+args.balance*(marginal*(marginal.log()+math.log(codes))).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tx,ty,tb=[v.to(device) for v in train]; vx,vy,vb=[v.to(device) for v in test]
        tl,_=model.encode(tx,.1,True); vl,_=model.encode(vx,.1,True)
        tz=model.deterministic_code(tl); vz=model.deterministic_code(vl)
        tc=tl.argmax(1).cpu(); vc=vl.argmax(1).cpu()
        q=model.q_head(vz); q_mse=F.mse_loss(q,vy).item(); action=(q.argmax(1)==vy.argmax(1)).float().mean().item()
        probe=probe_accuracy(vc,vy.cpu(),tc,ty.cpu())
        recon_logits=model.recon_head(vz).view(-1,args.length,len(VOCAB))
        recon=F.cross_entropy(recon_logits.reshape(-1,len(VOCAB)),vx.reshape(-1),ignore_index=0).item()
        used=len(vc.unique()); entropy=-(torch.bincount(vc,minlength=codes).float().div(len(vc)).clamp_min(1e-9)*torch.bincount(vc,minlength=codes).float().div(len(vc)).clamp_min(1e-9).log2()).sum().item()
    return {"objective":objective,"codes":codes,"bits":math.log2(codes),"seed":seed,"action_accuracy":action,"probe_accuracy":probe,"q_mse":q_mse,"reconstruction_mse":recon,"used_codes":used,"empirical_bits":entropy,"seconds":time.time()-start}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--train",type=int,default=6000); p.add_argument("--test",type=int,default=2000); p.add_argument("--epochs",type=int,default=25); p.add_argument("--batch",type=int,default=256); p.add_argument("--hidden",type=int,default=64); p.add_argument("--length",type=int,default=32); p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--balance",type=float,default=.05); p.add_argument("--codes",default="2,4,8"); p.add_argument("--objectives",default="q,reconstruction"); p.add_argument("--seeds",type=int,default=5); p.add_argument("--out",default="learned_results"); a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows=[]
    for codes in tuple(int(x) for x in a.codes.split(",")):
        for objective in tuple(x.strip() for x in a.objectives.split(",")):
            for seed in range(a.seeds):
                r=run_one(objective,codes,seed,a,device); rows.append(r); print(r,flush=True)
    os.makedirs(a.out,exist_ok=True)
    with open(os.path.join(a.out,"runs.csv"),"w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    summary={"device":str(device),"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"config":vars(a),"runs":rows}
    with open(os.path.join(a.out,"results.json"),"w") as f: json.dump(summary,f,indent=2)

if __name__=="__main__": main()
