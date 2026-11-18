from pathlib import Path
import json,random,math,hashlib,difflib
root=Path('.');rows=json.loads(next((root/'.cache').glob('train-*.json')).read_text(encoding='utf8'));rng=random.Random(20261117)
groups={l:[i for i,r in enumerate(rows) if r['label']==l] for l in (0,1)}
for a in groups.values():rng.shuffle(a)
(root/'data/subsets').mkdir(exist_ok=True)
for pct in (25,50):
 n=len(rows)*pct//100;raw={l:n*len(a)/len(rows) for l,a in groups.items()};counts={l:math.floor(v) for l,v in raw.items()}
 for l in sorted(groups,key=lambda l:(-(raw[l]-counts[l]),l))[:n-sum(counts.values())]:counts[l]+=1
 pos=sorted(i for l,a in groups.items() for i in a[:counts[l]])
 obj={'percent':pct,'sampling_seed':20261117,'size':n,'label_counts':counts,'positions':pos,'ids':[rows[i]['idx'] for i in pos]}
 (root/f'data/subsets/train_{pct}.json').write_text(json.dumps(obj,indent=2),encoding='utf8')
