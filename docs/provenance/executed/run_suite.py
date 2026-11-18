"""Run predeclared settings sequentially on one CUDA GPU."""
import json,subprocess,sys,platform,hashlib
from pathlib import Path
import torch,transformers,pyarrow
plan=[{'mode':m,'lr':2e-5 if m=='full' else 1e-3,'seeds':[17,42,73]} for m in ('full','head','bitfit','query','query_intermediate')]+[{'mode':'head','lr':lr,'seeds':[17,42,73]} for lr in (1e-4,4e-4)]
Path('results_local').mkdir(exist_ok=True)
Path('results_local/plan.json').write_text(json.dumps({'experiments':plan,'selection':'no checkpoint or hyperparameter selection; report all settings','epochs':3,'batch':8,'max_length':128,'attention':'eager'},indent=2),encoding='utf8')
env={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,'pyarrow':pyarrow.__version__,'gpu':torch.cuda.get_device_name(),'platform':platform.platform(),'bf16':torch.cuda.is_bf16_supported(),'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),'training_script_sha256':hashlib.sha256(Path('train.py').read_bytes()).hexdigest()}
Path('results_local/environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
for cfg in plan:
 print('CONFIG',cfg,flush=True)
 subprocess.run([sys.executable,'-s','-u','train.py','--mode',cfg['mode'],'--lr',str(cfg['lr'])],check=True)
print('ALL 21 RUNS COMPLETED',flush=True)
