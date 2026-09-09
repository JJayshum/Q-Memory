#!/usr/bin/env python3
"""Practical natural-language memory pilot with positional attention pooling."""
import argparse, json, math, random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

TRAIN=["The corridor sign says the {side} door is unlocked.","A note on the wall confirms: use the {side} exit.","The guard updates the log: the {side} gate is safe.","The panel flashes that the {side} passage is available."]
TEST=["The hallway display now marks the {side} doorway as open.","A fresh message reads: proceed through the {side} entrance.","The terminal reports that the {side} route is currently enabled."]
DIST=["You hear a distant fan.","The floor is made of old stone.","A small lamp flickers nearby.","Dust lies along the baseboard.","The room smells faintly of paper."]

def make(n,seed,templates,vocab=None,L=128):
    rng=random.Random(seed); texts=[]; ys=[]
    for _ in range(n):
        truth=rng.randrange(2); side="left" if truth==0 else "right"; s=["You stand before two identical doors."]
        s += [rng.choice(DIST) for _ in range(rng.randint(2,5))]+["Earlier, a faded note mentioned the other door."]
        s += [rng.choice(DIST) for _ in range(rng.randint(2,6))]+[rng.choice(templates).format(side=side)]
        s += [rng.choice(DIST) for _ in range(rng.randint(3,8))]+["choose open left door or open right door"]
        texts.append(" ".join(s).lower().replace("."," ").split()); ys.append(truth)
    if vocab is None:vocab={"<pad>":0}
    for t in texts:
        for w in t:
            if w not in vocab:vocab[w]=len(vocab)
    x=torch.zeros(n,L,dtype=torch.long)
    for i,t in enumerate(texts): x[i,:min(L,len(t))]=torch.tensor([vocab[w] for w in t[:L]])
    return x,torch.tensor(ys),vocab

class AttentionMemory(nn.Module):
    def __init__(self,vocab,codes,hidden=64,L=128):
        super().__init__(); self.emb=nn.Embedding(vocab,hidden,padding_idx=0); self.pos=nn.Parameter(torch.randn(1,L,hidden)*.02); self.score=nn.Linear(hidden,1); self.code=nn.Linear(hidden,codes)
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]; valid=x!=0; a=self.score(torch.tanh(h)).squeeze(-1).masked_fill(~valid,-1e9); w=F.softmax(a,1); pooled=(h*w.unsqueeze(-1)).sum(1); return self.code(pooled),pooled

def run(seed,a,device,objective):
    torch.manual_seed(seed); x,y,v=make(a.train,1000+seed,TRAIN); vx,vy,_=make(a.test,9000+seed,TEST,vocab=v); x,y,vx,vy=[z.to(device) for z in (x,y,vx,vy)]; m=AttentionMemory(len(v),2,a.hidden).to(device); opt=torch.optim.AdamW(m.parameters(),lr=a.lr)
    recon=nn.Linear(a.hidden,len(v)).to(device)
    opt=torch.optim.AdamW(list(m.parameters())+list(recon.parameters()),lr=a.lr)
    for _ in range(a.epochs):
        code,h=m(x)
        if objective=="q": loss=F.cross_entropy(code,y)
        elif objective=="reconstruction": loss=F.cross_entropy(recon(h).unsqueeze(1).expand(-1,x.size(1),-1).reshape(-1,len(v)),x.reshape(-1),ignore_index=0)
        else: loss=F.mse_loss(code[:,0],y.float())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        code,h=m(vx); test=(code.argmax(1)==vy).float().mean().item(); train=(m(x)[0].argmax(1)==y).float().mean().item()
    return {"objective":objective,"seed":seed,"train_accuracy":train,"accuracy":test,"device":str(device)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--train",type=int,default=6000); p.add_argument("--test",type=int,default=2000); p.add_argument("--seeds",type=int,default=5); p.add_argument("--epochs",type=int,default=300); p.add_argument("--hidden",type=int,default=64); p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--out",default="attention_memory_results"); a=p.parse_args(); d=torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows=[]
    for obj in ("q","reconstruction","reward"):
        for s in range(a.seeds): rows.append(run(s,a,d,obj)); print(rows[-1],flush=True)
    Path(a.out).mkdir(exist_ok=True); json.dump({"config":vars(a),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"rows":rows},open(Path(a.out)/"results.json","w"),indent=2)
if __name__=="__main__": main()
