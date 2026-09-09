import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

p='/root/autodl-tmp/modelscope_cache_7b/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master'
t=AutoTokenizer.from_pretrained(p)
m=AutoModelForCausalLM.from_pretrained(p,torch_dtype=torch.float16,device_map='auto')
x=t('Choose exactly one: open left door or open right door. The sign says the right door is unlocked. Answer with the action only.',return_tensors='pt').to('cuda')
y=m.generate(**x,max_new_tokens=8,do_sample=False)
print(t.decode(y[0],skip_special_tokens=True))
