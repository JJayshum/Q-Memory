#!/usr/bin/env python3
import argparse,json,math,random,statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

TRAIN_T=["The panel records route {s} as active.","A supervisor note confirms option {s}.","The terminal updates the current choice to {s}.","A radio message says select route {s}."]
TEST_T=["The display now marks path {s} as available.","A fresh dispatch instructs using route {s}.","The control console reports path {s} enabled."]
DIST=["The hallway is quiet.","A small light flickers.","Dust covers the floor.","The ceiling fan hums.","The room smells of paper."]
STATES=['red','blue','green','yellow']
def make(n,seed,templates,L=128,vocab=None):
 r=random.Random(seed); rows=[]
 for _ in range(n):
  y=r.randrange(4); s=STATES[y]; seq=['You face four identical buttons.']+[r.choice(DIST) for _ in range(r.randint(3,7))]+['Earlier, a note mentioned a different route.']+[r.choice(DIST) for _ in range(r.randint(3,8))]+[r.choice(templates).format(s=s)]+[r.choice(DIST) for _ in range(r.randint(4,10))]
  rows.append((' '.join(seq).lower().replace('.',' ').split(),y))
 if vocab is None:vocab={'<pad>':0}
 for t,_ in rows:
  for w in t:
   if w not in vocab:vocab[w]=len(vocab)
 x=torch.zeros(n,L,dtype=torch.long); y=torch.tensor([z[1] for z in rows])
 for i,(t,_) in enumerate(rows):x[i,:min(L,len(t))]=torch.tensor([vocab[w] for w in t[:L]])
 return x,y,vocab
class Continuous(nn.Module):
 def __init__(self,v,h):
  super().__init__();self.e=nn.Embedding(v,48,padding_idx=0);self.p=nn.Parameter(torch.randn(1,128,48)*.02);self.a=nn.Linear(48,1);self.h=nn.Linear(48,4)
 def forward(self,x):
  z=self.e(x)+self.p[:,:x.size(1)];m=x!=0;w=self.a(torch.tanh(z)).squeeze(-1).masked_fill(~m,-1e9);w=F.softmax(w,1);return self.h((z*w.unsqueeze(-1)).sum(1))
class Discrete(nn.Module):
 def __init__(self,v,c):
  super().__init__();self.e=nn.Embedding(v,48,padding_idx=0);self.p=nn.Parameter(torch.randn(1,128,48)*.02);self.a=nn.Linear(48,1);self.c=nn.Linear(48,c);self.h=nn.Linear(c,4,bias=False)
 def forward(self,x,tau=1,hard=True,stochastic=True):
  z=self.e(x)+self.p[:,:x.size(1)];m=x!=0;w=self.a(torch.tanh(z)).squeeze(-1).masked_fill(~m,-1e9);w=F.softmax(w,1);lg=self.c((z*w.unsqueeze(-1)).sum(1));q=F.gumbel_softmax(lg,tau=tau,hard=hard) if stochastic else F.softmax(lg/tau,dim=-1);return lg,self.h(q)
def run(kind,seed,a,d):
 torch.manual_seed(seed);x,y,v=make(a.train,1000+seed,TRAIN_T);vx,vy,_=make(a.test,9000+seed,TEST_T,vocab=v);x,y,vx,vy=[z.to(d) for z in (x,y,vx,vy)];
 if kind=='continuous':m=Continuous(len(v),a.hidden).to(d)
 else:m=Discrete(len(v),int(kind)).to(d)
 o=torch.optim.AdamW(m.parameters(),lr=a.lr)
 for ep in range(a.epochs):
  if kind=='continuous':loss=F.cross_entropy(m(x),y)
  else:
   lg,p=m(x,max(.35,1.5*(1-ep/a.epochs)),True,stochastic=False);loss=F.cross_entropy(lg,y) if int(kind)==4 else F.cross_entropy(p,y)
   if a.balance: z=F.softmax(lg,1).mean(0).clamp_min(1e-8);loss+=a.balance*(z*(z.log()+math.log(int(kind)))).sum()
  o.zero_grad();loss.backward();o.step()
 with torch.no_grad():
  if kind=='continuous':tr=(m(x).argmax(1)==y).float().mean().item();te=(m(vx).argmax(1)==vy).float().mean().item();used=1
  else:lg,p=m(vx,.1,True);tr=(m(x,.1,True)[0].argmax(1)==y).float().mean().item() if int(kind)==4 else (m(x,.1,True)[1].argmax(1)==y).float().mean().item();te=(lg.argmax(1)==vy).float().mean().item() if int(kind)==4 else (p.argmax(1)==vy).float().mean().item();used=int(lg.argmax(1).unique().numel())
 return {'kind':kind,'seed':seed,'train_accuracy':tr,'accuracy':te,'used_codes':used}
def main():
 p=argparse.ArgumentParser();p.add_argument('--train',type=int,default=8000);p.add_argument('--test',type=int,default=3000);p.add_argument('--epochs',type=int,default=250);p.add_argument('--seeds',type=int,default=5);p.add_argument('--hidden',type=int,default=64);p.add_argument('--lr',type=float,default=2e-3);p.add_argument('--balance',type=float,default=0);p.add_argument('--out',default='multistate_results');a=p.parse_args();d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[]
 for k in ('continuous','1','4'):
  for s in range(a.seeds):rows.append(run(k,s,a,d));print(rows[-1],flush=True)
 Path(a.out).mkdir(exist_ok=True);json.dump({'config':vars(a),'device':str(d),'rows':rows},open(Path(a.out,'results.json'),'w'),indent=2)
if __name__=='__main__':main()
