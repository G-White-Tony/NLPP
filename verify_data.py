"""Independent verification of saved local experiments (no CUDA needed)."""
import csv,json,math,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean,stdev
import pyarrow.parquet as pq

def load(p):return json.loads(p.read_text(encoding='utf8'))
def verify():
 root=Path('results_data');manifest=load(Path('data/resources.json'))
 for f in manifest:
  p=Path('.cache')/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],str(p)
 rows=pq.read_table('.cache/validation-00000-of-00001.parquet').to_pylist()
 assert [r['idx'] for r in rows]==load(Path('data/validation_ids.json'))
 expected={r['idx']:r for r in rows};summaries=[];heads=defaultdict(set)
 train=load(next(Path('.cache').glob('train-*.json')))
 subsets={pct:load(Path(f'data/subsets/train_{pct}.json')) for pct in (25,50)}
 assert set(subsets[25]['ids']) < set(subsets[50]['ids']) <= {r['idx'] for r in train}
 assert set(subsets[50]['ids']).isdisjoint(expected)
 for pct,sub in subsets.items():
  assert len(sub['positions'])==len(set(sub['positions']))==sub['size']==3668*pct//100
  assert sub['ids']==[train[i]['idx'] for i in sub['positions']]
  for label in (0,1):
   assert sub['label_counts'][str(label)]==sum(train[i]['label']==label for i in sub['positions'])
   assert abs(sub['label_counts'][str(label)]-sum(r['label']==label for r in train)*pct/100)<=1
 env=load(root/'environment.json')
 assert env['nvidia_smi']==load(Path('results_local/environment.json'))['nvidia_smi']
 assert hashlib.sha256(Path('train_data.py').read_bytes()).hexdigest()==env['training_script_sha256']
 counts={'full':109483778,'head':1538,'bitfit':104450,'query':10754,'query_intermediate':47618}
 for mp in sorted(root.glob('[0-9]*/*/metrics.json')):
  m=load(mp);p=mp.parent;preds=[json.loads(x) for x in (p/'predictions.jsonl').read_text(encoding='utf8').splitlines()];a=load(p/'audit.json');mode=m['mode'];seed=m['seed']
  assert len(preds)==408 and [r['idx'] for r in preds]==[r['idx'] for r in rows]
  for r in preds:
   assert r['label']==expected[r['idx']]['label'];assert 0<=r['p1']<=1;assert r['prediction'] in (0,1)
   # Probability rounding may create a printed tie; compare away from ties.
   if abs(r['p1']-.5)>1e-6:assert r['prediction']==int(r['p1']>.5)
  cm=Counter((r['label'],r['prediction']) for r in preds);tn,fp,fn,tp=(cm[k] for k in ((0,0),(0,1),(1,0),(1,1)))
  f1=2*tp/(2*tp+fp+fn);acc=(tp+tn)/408
  assert math.isclose(m['f1'],f1) and math.isclose(m['accuracy'],acc)
  sub=subsets[m['percent']];size=sub['size'];per_epoch=math.ceil(size/8)
  target=1377 if m['budget']=='steps' else 3*per_epoch
  assert m['train_examples']==size and m['steps']==target and m['epochs']==math.ceil(target/per_epoch)
  assert len(m['losses'])==len(m['epoch_examples'])==m['epochs']
  remaining=target;seen=[]
  while remaining:
   steps_this=min(remaining,per_epoch);seen.append(min(size,steps_this*8));remaining-=steps_this
  assert m['epoch_examples']==seen and m['examples_seen']==sum(seen)
  assert m['subset_sha256']==hashlib.sha256(Path(m['subset_path']).read_bytes()).hexdigest()
  assert m['training_script_sha256']==env['training_script_sha256']
  base=load(Path('results_local')/str(seed)/(mode+('_lr2e-05' if mode=='full' else '_lr1e-03'))/'metrics.json')
  for k in ('initial_head_sha256','torch','transformers','python','attention','precision','lr','batch','gpu'):
   assert m[k]==base[k],(seed,mode,k)
  assert m['total']==109483778 and m['trainable']==counts[mode] and m['optimizer_group_verified']
  assert len(a)==len({x['name'] for x in a})==201
  assert sum(x['numel'] for x in a if x['trainable'])==counts[mode]
  for x in a:
   n=x['name'];allowed=mode=='full' or n.startswith('classifier.') or (mode=='bitfit' and n.endswith('.bias')) or (mode=='query' and n.endswith('.attention.self.query.bias')) or (mode=='query_intermediate' and (n.endswith('.attention.self.query.bias') or n.endswith('.intermediate.dense.bias')))
   assert x['trainable']==allowed
   if allowed:assert x['first_grad_norm'] is not None and math.isfinite(x['first_grad_norm']) and x['first_grad_norm']>0 and 0<x['changed_elements']<=x['numel'] and x['max_abs_delta']>0
   else:assert x['first_grad_norm'] is None and x['changed_elements']==0 and x['max_abs_delta']==0
  heads[seed].add(m['initial_head_sha256'])
  summaries.append(dict(seed=seed,config=m['config_id'],mode=mode,lr=m['lr'],percent=m['percent'],budget=m['budget'],train_examples=m['train_examples'],steps=m['steps'],examples_seen=m['examples_seen'],F1=f1*100,Accuracy=acc*100,positive_rate=(tp+fp)/408*100,negative_recall=tn/(tn+fp)*100,balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))*50,trainable=m['trainable'],trainable_pct=m['trainable']/m['total']*100,updated_tensors=sum(x['changed_elements']>0 for x in a),changed_elements=sum(x['changed_elements'] for x in a),peak_MiB=m['peak_allocated_MiB'],seconds=m['seconds'],TN=tn,FP=fp,FN=fn,TP=tp))
 assert len(summaries)==18,len(summaries)
 assert all(len(v)==1 for v in heads.values()) and set(heads)=={17,42,73}
 with (root/'summary_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
 groups={}
 for cfg in sorted({r['config'] for r in summaries}):
  rr=[r for r in summaries if r['config']==cfg];assert len(rr)==3
  g={k:{'mean':mean(r[k] for r in rr),'sd':stdev(r[k] for r in rr)} for k in ('F1','Accuracy','positive_rate','negative_recall','balanced_accuracy','peak_MiB','seconds')}
  g.update(config=cfg,mode=rr[0]['mode'],lr=rr[0]['lr'],percent=rr[0]['percent'],budget=rr[0]['budget'],train_examples=rr[0]['train_examples'],steps=rr[0]['steps'],examples_seen=rr[0]['examples_seen'],trainable=rr[0]['trainable'],trainable_pct=rr[0]['trainable_pct']);groups[cfg]=g
 (root/'aggregate.json').write_text(json.dumps(groups,indent=2),encoding='utf8')
 report={'verified_runs':18,'nested_stratified_subsets_verified':True,'actual_examples_and_optimizer_steps_verified':True,'same_initialization_and_environment_as_full_data':True,'all_resource_hashes_verified':True,'all_predictions_checked_against_source_labels':True,'all_frozen_parameters_unchanged':True,'optimizer_groups_verified_in_training':True,'same_seed_initial_head_hashes_match':True,'total_formal_training_seconds':sum(r['seconds'] for r in summaries)}
 (root/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
 for k,g in groups.items():print(k,'F1',g['F1'],'Accuracy',g['Accuracy'],'positive_rate',g['positive_rate'])
 return summaries
if __name__=='__main__':verify()
