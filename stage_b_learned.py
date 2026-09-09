#!/usr/bin/env python3
"""Small learned-memory pilot over Stage B replay labels."""
import argparse, csv, json, math, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def load_data(path):
    rows=json.loads(Path(path).read_text()); vocab={"<pad>":0}; samples=[]
    for row in rows:
        for policy, vals in row["q"].items():
            if policy != "oracle_success": continue
            for action,v in vals.items():
                text=(row.get("observation","")+" "+action).lower().split()
                ids=[]
                for tok in text:
                    if tok not in vocab: vocab[tok]=len(vocab)
                    ids.append(vocab[tok])
                # Centered action advantage supplies dense positive/negative
                # supervision even when terminal rewards are sparse.
                samples.append({"ids":ids,"y":float(v["advantage"]),"split":row.get("split","test"),"game":row["game"]})
    L=max(len(s["ids"]) for s in samples); x=torch.zeros(len(samples),L,dtype=torch.long)
    y=torch.tensor([s["y"] for s in samples],dtype=torch.float32)
    for i,s in enumerate(samples): x[i,:len(s["ids"])]=torch.tensor(s["ids"])
    train=torch.tensor([s["split"]=="train" for s in samples]); test=~train
    return x,y,train,test,vocab

class Bottleneck(nn.Module):
    def __init__(self,vocab,codes,hidden=48):
        super().__init__(); self.emb=nn.Embedding(vocab,24,padding_idx=0); self.gru=nn.GRU(24,hidden,batch_first=True)
        self.to_code=nn.Linear(hidden,codes); self.head=nn.Linear(codes,1)
    def forward(self,x,tau=1.0,hard=True):
        _,h=self.gru(self.emb(x)); logits=self.to_code(h[-1]); z=F.gumbel_softmax(logits,tau=tau,hard=hard); return logits,z

def run(codes,seed,args,device):
    torch.manual_seed(seed); x,y,tr,te,vocab=load_data(args.labels); x,y=x.to(device),y.to(device); tr,te=tr.to(device),te.to(device)
    m=Bottleneck(len(vocab),codes,args.hidden).to(device); opt=torch.optim.AdamW(m.parameters(),lr=args.lr)
    for ep in range(args.epochs):
        tau=max(.35,1.5*(1-ep/max(1,args.epochs))); logits,z=m(x,tau,True); pred=m.head(z).squeeze(-1); loss=F.smooth_l1_loss(pred[tr],y[tr])
        marginal=F.softmax(logits[tr]/tau,-1).mean(0).clamp_min(1e-8); loss += args.balance*(marginal*(marginal.log()+math.log(codes))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval();
    with torch.no_grad():
        lg,z=m(x,.1,True); code=lg.argmax(-1); pred=m.head(F.one_hot(code,codes).float()).squeeze(-1)
    p=pred[te]; truth=y[te]; mse=F.mse_loss(p,truth).item(); code_acc={}
    test_code = code[te]
    for c in test_code.unique().tolist(): code_acc[c]=float(truth[test_code==c].mean())
    baseline=float(y[tr].mean()); const=F.mse_loss(torch.full_like(truth,baseline),truth).item()
    return {"codes":codes,"seed":seed,"test_mse":mse,"constant_test_mse":const,"improvement":const-mse,"test_n":int(te.sum()),"used_codes":int(code[te].unique().numel()),"device":str(device)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--labels",default="stage_b_medium2/labels.json"); p.add_argument("--codes",default="2,4,8"); p.add_argument("--seeds",type=int,default=3); p.add_argument("--epochs",type=int,default=150); p.add_argument("--hidden",type=int,default=48); p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--balance",type=float,default=.05); p.add_argument("--out",default="stage_b_learned_results"); a=p.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows=[]
    for c in map(int,a.codes.split(",")):
        for s in range(a.seeds): rows.append(run(c,s,a,device)); print(rows[-1],flush=True)
    Path(a.out).mkdir(exist_ok=True); json.dump({"config":vars(a),"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"runs":rows},open(Path(a.out)/"results.json","w"),indent=2); csv.DictWriter(open(Path(a.out)/"runs.csv","w",newline=""),fieldnames=rows[0]).writeheader()
if __name__=="__main__": main()
