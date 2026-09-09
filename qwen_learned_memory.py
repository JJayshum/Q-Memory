import json,random,torch
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
MODEL='/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master'
DIST=['The corridor is quiet.','A lamp flickers.','Dust covers the floor.','You hear a distant fan.']
def ep(seed):
 r=random.Random(seed); y=r.randrange(2); side='left' if y==0 else 'right'; s=['You face two identical doors.']+[r.choice(DIST) for _ in range(12)]+['Earlier, a note said the other door was safe.']+[r.choice(DIST) for _ in range(12)]+[f'A fresh security update says the {side} door is unlocked.']+[r.choice(DIST) for _ in range(12)]; return ' '.join(s),y,side
def build(n,start,vocab=None,L=128):
 rows=[ep(start+i) for i in range(n)]; vocab={} if vocab is None else vocab; toks=[]
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
 messages=[{'role':'system','content':'You are an action selector. Reply with exactly one of: open left door; open right door.'},{'role':'user','content':p}]
 prompt=t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); x=t(prompt,return_tensors='pt').to('cuda'); y=m.generate(**x,max_new_tokens=16,do_sample=False); return t.decode(y[0][x.input_ids.shape[1]:],skip_special_tokens=True).lower()
def main():
 torch.manual_seed(0); x,train_y,v,_=build(6000,1000); m=Mem(len(v)).cuda(); o=torch.optim.AdamW(m.parameters(),lr=2e-3)
 for _ in range(300): loss=F.cross_entropy(m(x.cuda()),train_y.cuda()); o.zero_grad(); loss.backward(); o.step()
 t=AutoTokenizer.from_pretrained(MODEL); lm=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,device_map='auto'); rows=[]
 for i in range(100):
  s,label,side=ep(9000+i); ids=torch.tensor([v.get(w,0) for w in s.lower().replace('.',' ').split()[:128]])[None].cuda(); pred=int(m(ids).argmax(1)); learned='left' if pred==0 else 'right'; short=' '.join(s.split()[-12:]); outs={'full':ask(lm,t,s),'truncated':ask(lm,t,short),'learned':ask(lm,t,f'Use compressed memory. The correct action is to open the {learned} door.')}; rows.append({'truth':side,'predicted_memory':learned,'outputs':outs,'correct':{k:(side in z and ('left' if side=='left' else 'right') in z) for k,z in outs.items()}})
 acc={k:sum(r['correct'][k] for r in rows)/100 for k in ('full','truncated','learned')}; Path('/root/autodl-tmp/qwen_learned_results').mkdir(exist_ok=True); json.dump({'episodes':100,'accuracy':acc,'memory_train_accuracy':float((m(x.cuda()).argmax(1)==train_y.cuda()).float().mean()),'rows':rows},open('/root/autodl-tmp/qwen_learned_results/results.json','w'),indent=2); print(acc)
if __name__=='__main__':main()
