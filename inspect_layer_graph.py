"""Inspect one-batch autograd connectivity; separate from formal timing."""
from pathlib import Path
import json,hashlib
import torch
from transformers import BertForSequenceClassification,BertTokenizerFast,set_seed
from train_layers import dataset,LAYERS
import re
rows,features=dataset('train',BertTokenizerFast.from_pretrained('.cache/bert'))
results={}
for mode in ('bottom','top','alternating'):
 set_seed(17);model=BertForSequenceClassification.from_pretrained('.cache/bert',attn_implementation='eager').cuda();model.train()
 for name,p in model.named_parameters():
  m=re.match(r'bert\.encoder\.layer\.(\d+)\.',name)
  p.requires_grad_(name.startswith('classifier.') or (m is not None and int(m.group(1)) in LAYERS[mode] and name.endswith('.bias')))
 records=[];hooks=[]
 def record(layer):
  def hook(mod,args,output):
   t=output[0];records.append({'layer_zero_based':layer,'output_requires_grad':t.requires_grad,'output_has_grad_fn':t.grad_fn is not None})
  return hook
 for i,layer in enumerate(model.bert.encoder.layer):hooks.append(layer.register_forward_hook(record(i)))
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**{k:v[:8] for k,v in features.items()}).loss
 loss.backward()
 for h in hooks:h.remove()
 expected=[i>=6 if mode=='top' else True for i in range(12)]
 assert [r['output_requires_grad'] for r in records]==expected
 results[mode]={'batch_size':8,'seed':17,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'layer_outputs':records}
 del model
Path('results_layers/graph_probe.json').write_text(json.dumps({'purpose':'single-batch autograd connectivity, outside formal timing; not an operator-level memory profile','torch':torch.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results},indent=2),encoding='utf8')
for mode,r in results.items():print(mode,[x['output_requires_grad'] for x in r['layer_outputs']])
