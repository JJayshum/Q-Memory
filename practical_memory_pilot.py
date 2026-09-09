#!/usr/bin/env python3
"""Natural-language, action-symmetric memory benchmark with held-out templates."""
import argparse, json, math, random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

TRAIN_TEMPLATES=[
 "The corridor sign says the {side} door is unlocked.",
 "A note on the wall confirms: use the {side} exit.",
 "The guard updates the log: the {side} gate is safe.",
 "The panel flashes that the {side} passage is available.",
]
TEST_TEMPLATES=[
 "The hallway display now marks the {side} doorway as open.",
 "A fresh message reads: proceed through the {side} entrance.",
 "The terminal reports that the {side} route is currently enabled.",
]
DISTRACT=["You hear a distant fan.","The floor is made of old stone.","A small lamp flickers nearby.","Dust lies along the baseboard.","The room smells faintly of paper."]

def make_data(n,seed,templates,length=128,vocab=None):
    rng=random.Random(seed); rows=[]
    for _ in range(n):
        truth=rng.randrange(2); side="left" if truth==0 else "right"
        seq=["You stand before two identical doors."]
        seq += [rng.choice(DISTRACT) for _ in range(rng.randint(2,5))]
        seq += ["Earlier, a faded note mentioned the other door."]
        seq += [rng.choice(DISTRACT) for _ in range(rng.randint(2,6))]
        seq += [rng.choice(templates).format(side=side)]
        seq += [rng.choice(DISTRACT) for _ in range(rng.randint(3,8))]
        text=" ".join(seq)+" choose open left door or open right door"
        rows.append((text,truth))
    if vocab is None: vocab={"<pad>":0}
    toks=[]
    for text,_ in rows:
        ws=text.lower().replace("."," ").split(); toks.append(ws)
        for w in ws:
            if w not in vocab:vocab[w]=len(vocab)
    L=min(length,max(len(t) for t in toks)); x=torch.zeros(n,L,dtype=torch.long); y=torch.tensor([r[1] for r in rows])
    for i,t in enumerate(toks): x[i,:min(L,len(t))]=torch.tensor([vocab[w] for w in t[:L]])
    return x,y,vocab

class Memory(nn.Module):
    def __init__(self,vocab,codes,hidden):
        super().__init__(); self.emb=nn.Embedding(vocab,32,padding_idx=0); self.gru=nn.GRU(32,hidden,batch_first=True); self.to_code=nn.Linear(hidden,codes); self.head=nn.Linear(codes,2,bias=False)
    def forward(self,x,tau=1,hard=True,stochastic=True):
        out,_=self.gru(self.emb(x)); last=(x!=0).sum(1).clamp_min(1)-1; h=out[torch.arange(x.size(0),device=x.device),last]; lg=self.to_code(h)
        if stochastic: z=F.gumbel_softmax(lg,tau=tau,hard=hard)
        else: z=F.softmax(lg/tau,-1)
        return lg,self.head(z)

class Continuous(nn.Module):
    def __init__(self,vocab,hidden):
        super().__init__(); self.emb=nn.Embedding(vocab,32,padding_idx=0); self.gru=nn.GRU(32,hidden,batch_first=True); self.head=nn.Linear(hidden,2)
    def forward(self,x):
        out,_=self.gru(self.emb(x)); last=(x!=0).sum(1).clamp_min(1)-1; h=out[torch.arange(x.size(0),device=x.device),last]; return self.head(h)

class DirectCode(nn.Module):
    def __init__(self,vocab,codes,hidden):
        super().__init__(); self.emb=nn.Embedding(vocab,32,padding_idx=0); self.gru=nn.GRU(32,hidden,batch_first=True); self.to_code=nn.Linear(hidden,codes)
    def forward(self,x):
        out,_=self.gru(self.emb(x)); last=(x!=0).sum(1).clamp_min(1)-1; h=out[torch.arange(x.size(0),device=x.device),last]; return self.to_code(h)

def run(codes,seed,a,device):
    torch.manual_seed(seed); x,y,vocab=make_data(a.train,1000+seed,TRAIN_TEMPLATES); test_templates=TRAIN_TEMPLATES if a.in_distribution else TEST_TEMPLATES; vx,vy,_=make_data(a.test,9000+seed,test_templates,vocab=vocab); x,y,vx,vy=[z.to(device) for z in (x,y,vx,vy)]
    direct=a.direct_code and codes==2
    if direct and a.pretrain:
        base=Continuous(len(vocab),a.hidden).to(device); bo=torch.optim.AdamW(base.parameters(),lr=a.lr)
        for _ in range(a.pretrain):
            bp=base(x); bl=F.cross_entropy(bp,y); bo.zero_grad(); bl.backward(); bo.step()
        m=DirectCode(len(vocab),codes,a.hidden).to(device); m.emb.load_state_dict(base.emb.state_dict()); m.gru.load_state_dict(base.gru.state_dict()); m.to_code.load_state_dict(base.head.state_dict())
    else: m=Continuous(len(vocab),a.hidden).to(device) if codes==0 else (DirectCode(len(vocab),codes,a.hidden).to(device) if direct else Memory(len(vocab),codes,a.hidden).to(device))
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr)
    for ep in range(a.epochs):
        if codes==0 or direct: p=m(x); loss=F.cross_entropy(p,y)
        else:
            tau=max(.35,1.5*(1-ep/a.epochs)); lg,p=m(x,tau,True,stochastic=not a.soft_train); loss=F.cross_entropy(p,y); marginal=F.softmax(lg/tau,-1).mean(0).clamp_min(1e-8); loss+=a.balance*(marginal*(marginal.log()+math.log(codes))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        if codes==0: p=m(vx); train_p=m(x); code=torch.zeros(len(vy),device=device,dtype=torch.long)
        elif direct: p=m(vx); train_p=m(x); code=p.argmax(1)
        else: lg,p=m(vx,.1,True); code=lg.argmax(1); train_p=m(x,.1,True)[1]
        acc=(p.argmax(1)==vy).float().mean().item(); train_acc=(train_p.argmax(1)==y).float().mean().item()
    return {"codes":codes,"seed":seed,"accuracy":acc,"train_accuracy":train_acc,"used_codes":int(code.unique().numel()),"device":str(device)}

def baselines(a):
    # Action-only is exactly chance because candidate names are symmetric.
    return {"action_only":0.5,"random":0.5}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--train",type=int,default=6000); p.add_argument("--test",type=int,default=2000); p.add_argument("--codes",default="1,2,4,8"); p.add_argument("--seeds",type=int,default=5); p.add_argument("--epochs",type=int,default=150); p.add_argument("--pretrain",type=int,default=0); p.add_argument("--hidden",type=int,default=64); p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--balance",type=float,default=.05); p.add_argument("--soft-train",action="store_true"); p.add_argument("--direct-code",action="store_true"); p.add_argument("--in-distribution",action="store_true"); p.add_argument("--out",default="practical_memory_results"); a=p.parse_args(); d=torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows=[]
    for c in map(int,a.codes.split(",")):
        for s in range(a.seeds): rows.append(run(c,s,a,d)); print(rows[-1],flush=True)
    Path(a.out).mkdir(exist_ok=True); json.dump({"config":vars(a),"device":str(d),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"baselines":baselines(a),"rows":rows},open(Path(a.out)/"results.json","w"),indent=2)
if __name__=="__main__": main()
