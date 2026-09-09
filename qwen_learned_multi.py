import json, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL='/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master'
DIST=['The corridor is quiet.','A lamp flickers.','Dust covers the floor.','You hear a distant fan.']
def episode(seed):
 r=random.Random(seed); y=r.randrange(2); side='left' if y==0 else 'right'; s=['You face two identical doors.']+[r.choice(DIST) for _ in range(12)]+['Earlier, a note said the other door was safe.']+[r.choice(DIST) for _ in range(12)]+[f'A fresh security update says the {side} door is unlocked.']+[r.choice(DIST) for _ in range(12)]; return ' '.join(s),y,side
def build(n,start,vocab=None,L=128):
 rows=[episode(start+i) for i in range(n)]; vocab={} if vocab is None else vocab; toks=[]
 for s,_,_ in rows:
  t=s.lower().replace('.',' ').split(); toks.append(t)
  for w in t:
   if w not in vocab:vocab[w]=len(vocab)+1
 x=torch.zeros(n,L,dtype=torch.long); y=torch.tensor([r[1] for r in rows])
 for i,t in enumerate(toks): x[i,:min(L,len(t))]=torch.tensor([vocab[w] for w in t[:L]])
 return x,y,vocab,rows
class Mem(nn.Module):
 def __init__(self,v):
  super().__init__(); self.e=nn.Embedding(v+1,48,padding_idx=0); self.p=nn.Parameter(torch.randn(1,128,48)*.02); self.a=nn.Linear(48,1); self.h=nn.Linear(48,2)
 def forward(self,x):
  z=self.e(x)+self.p[:,:x.size(1)]; m=x!=0; w=self.a(torch.tanh(z)).squeeze(-1).masked_fill(~m,-1e9); w=F.softmax(w,1); h=(z*w.unsqueeze(-1)).sum(1); return self.h(h)
def ask(m,t,p):
 msg=[{'role':'system','content':'You are an action selector. Reply with exactly one of: open left door; open right door.'},{'role':'user','content':p}]; q=t.apply_chat_template(msg,tokenize=False,add_generation_prompt=True); x=t(q,return_tensors='pt').to('cuda'); y=m.generate(**x,max_new_tokens=16,do_sample=False); return t.decode(y[0][x.input_ids.shape[1]:],skip_special_tokens=True).lower()
def train_mem(seed,vocab):
 torch.manual_seed(seed); x,y,_,_=build(6000,1000,vocab); m=Mem(len(vocab)).cuda(); o=torch.optim.AdamW(m.parameters(),lr=2e-3)
 for _ in range(300): loss=F.cross_entropy(m(x.cuda()),y.cuda()); o.zero_grad(); loss.backward(); o.step()
 return m,x,y
def main():
 tok=AutoTokenizer.from_pretrained(MODEL); lm=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,device_map='auto'); allrows=[]; started=time.time()
 for ms in range(3):
  _,_,v,_=build(6000,1000+ms*100,vocab=None); mem,tx,ty=train_mem(ms,v); mem.eval()
  for i in range(50):
   s,_,side=episode(9000+ms*100+i); ids=torch.tensor([v.get(w,0) for w in s.lower().replace('.',' ').split()[:128]])[None].cuda(); learned='left' if int(mem(ids).argmax(1))==0 else 'right'; short=' '.join(s.split()[-12:]); outs={'full':ask(lm,tok,s),'truncated':ask(lm,tok,short),'learned':ask(lm,tok,f'Use compressed memory. The correct action is to open the {learned} door.')}; allrows.append({'memory_seed':ms,'truth':side,'predicted_memory':learned,'correct':{k:(side in z and ('left' if side=='left' else 'right') in z) for k,z in outs.items()}})
  print('memory_seed',ms,'done',flush=True)
 acc={k:sum(r['correct'][k] for r in allrows)/len(allrows) for k in ('full','truncated','learned')}; out={'episodes':len(allrows),'accuracy':acc,'rows':allrows,'seconds':time.time()-started}; Path('/root/autodl-tmp/qwen_learned_multi_results').mkdir(exist_ok=True); json.dump(out,open('/root/autodl-tmp/qwen_learned_multi_results/results.json','w'),indent=2); print(acc)
if __name__=='__main__':main()
