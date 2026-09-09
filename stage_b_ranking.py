#!/usr/bin/env python3
"""Snapshot-grouped action ranking with observation-only discrete memory."""
import argparse, json, math, statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def tokenize(s): return s.lower().replace("-="," " ).replace("=-"," " ).split()

def load(path):
    raw=json.loads(Path(path).read_text()); vocab={"<pad>":0}; rows=[]
    for r in raw:
        obs=tokenize(r.get("observation","")); acts=[tokenize(a) for a in r["actions"]]
        for t in obs + [t for a in acts for t in a]:
            if t not in vocab: vocab[t]=len(vocab)
        q=r["q"]["oracle_success"]; y=[float(q[a]["advantage"]) for a in r["actions"]]
        rows.append({"obs":[vocab[t] for t in obs],"acts":[[vocab[t] for t in a] for a in acts],"y":y,"split":r.get("split","test")})
    ol=max(len(r["obs"]) for r in rows); al=max(len(a) for r in rows for a in r["acts"]); na=max(len(r["acts"]) for r in rows)
    obs=torch.zeros(len(rows),ol,dtype=torch.long); acts=torch.zeros(len(rows),na,al,dtype=torch.long); y=torch.full((len(rows),na),-1e9); mask=torch.zeros(len(rows),na,dtype=torch.bool)
    for i,r in enumerate(rows):
        obs[i,:len(r["obs"])]=torch.tensor(r["obs"]);
        for j,a in enumerate(r["acts"]): acts[i,j,:len(a)]=torch.tensor(a); y[i,j]=r["y"][j]; mask[i,j]=True
    tr=torch.tensor([r["split"]=="train" for r in rows]); return obs,acts,y,mask,tr,~tr,len(vocab)

class Ranker(nn.Module):
    def __init__(self,vocab,codes,hidden):
        super().__init__(); self.codes=codes; self.emb=nn.Embedding(vocab,32,padding_idx=0); self.gru=nn.GRU(32,hidden,batch_first=True); self.to_code=nn.Linear(hidden,codes); self.action=nn.Linear(32,codes,bias=False)
    def forward(self,obs,acts,tau=1,hard=True):
        _,h=self.gru(self.emb(obs)); logits=self.to_code(h[-1]); z=F.gumbel_softmax(logits,tau=tau,hard=hard)
        ae=self.emb(acts); am=(acts!=0).unsqueeze(-1); pooled=(ae*am).sum(2)/am.sum(2).clamp_min(1); return logits,(self.action(pooled)*z[:,None]).sum(-1)

def evaluate(scores,y,mask,sel):
    s=scores[sel].masked_fill(~mask[sel],-1e9); q=y[sel]; pred=s.argmax(1); best=q.argmax(1); top=(pred==best).float().mean().item(); chosen=q.gather(1,pred[:,None]).squeeze(1); optimal=q.max(1).values; regret=(optimal-chosen).mean().item(); positive=(optimal>0); pos_top=(pred[positive]==best[positive]).float().mean().item() if positive.any() else float("nan")
    return top,regret,pos_top,int(sel.sum())

def run(codes,seed,args,device):
    torch.manual_seed(seed); obs,acts,y,mask,tr,te,nv=load(args.labels); obs,acts,y,mask,tr,te=[x.to(device) for x in (obs,acts,y,mask,tr,te)]
    m=Ranker(nv,codes,args.hidden).to(device); opt=torch.optim.AdamW(m.parameters(),lr=args.lr)
    for ep in range(args.epochs):
        tau=max(.35,1.5*(1-ep/args.epochs)); lg,s=m(obs,acts,tau,True); informative=tr & (y.max(1).values>0); st=s[informative].masked_fill(~mask[informative],-1e9); target=y[informative].argmax(1); loss=F.cross_entropy(st,target)
        if codes > 1:
            marginal=F.softmax(lg[informative]/tau,-1).mean(0).clamp_min(1e-8); loss+=args.balance*(marginal*(marginal.log()+math.log(codes))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval();
    with torch.no_grad(): lg,s=m(obs,acts,.1,True); z=lg.argmax(1); informative_test=te & (y.max(1).values>0); top,regret,pos_top,n=evaluate(s,y,mask,informative_test); base=torch.zeros_like(s); btop,bregret,bpos,_=evaluate(base,y,mask,informative_test)
    return {"codes":codes,"seed":seed,"top1":top,"regret":regret,"positive_top1":pos_top,"lexicographic_top1":btop,"lexicographic_regret":bregret,"test_snapshots":n,"used_codes":int(z[informative_test].unique().numel()),"device":str(device)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--labels",required=True); p.add_argument("--codes",default="1,2,4,8"); p.add_argument("--seeds",type=int,default=5); p.add_argument("--epochs",type=int,default=250); p.add_argument("--hidden",type=int,default=64); p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--balance",type=float,default=.03); p.add_argument("--out",default="stage_b_ranking"); a=p.parse_args(); d=torch.device("cuda" if torch.cuda.is_available() else "cpu"); out=[]
    for c in map(int,a.codes.split(",")):
        for s in range(a.seeds): out.append(run(c,s,a,d)); print(out[-1],flush=True)
    Path(a.out).mkdir(exist_ok=True); json.dump({"config":vars(a),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"runs":out},open(Path(a.out)/"results.json","w"),indent=2)
if __name__=="__main__": main()
