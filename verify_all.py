"""Independently recheck all 48 local runs and regenerate the shared index."""
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import verify_local,verify_data,verify_layers

def verify():
 print('Verification UTC:',datetime.now(timezone.utc).isoformat())
 groups=[('results_local',verify_local.verify()),('results_data',verify_data.verify()),('results_layers',verify_layers.verify())]
 rows=[];proofs={};published={}
 for directory,records in groups:
  proofs[directory]=json.loads((Path(directory)/'verification.json').read_text(encoding='utf8'))
  for row in records:
   row['source_directory']=directory
   if directory!='results_data':row.update(percent=100,budget='epochs',train_examples=3668,steps=1377,examples_seen=11004)
   rows.append(row)
 for directory,name in [('results_local','train_local.py'),('results_data','train_data.py'),('results_layers','train_layers.py')]:
  expected=json.loads((Path(directory)/'environment.json').read_text(encoding='utf8'))['training_script_sha256']
  actual=hashlib.sha256(Path(name).read_bytes()).hexdigest();assert actual==expected,(directory,name)
  published[name]=actual
 assert len(rows)==len({(r['source_directory'],r['seed'],r['config']) for r in rows})==48
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path('all_local_runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 report={'verified_local_runs':48,'previous_local':proofs['results_local'],'new_data_experiments':proofs['results_data'],'new_layer_experiments':proofs['results_layers'],'published_training_files_sha256':published}
 Path('verification_all.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print('PASS: 48 formal local runs; exact training source hashes, raw predictions, labels and audits verified.')
 return report
if __name__=='__main__':verify()
