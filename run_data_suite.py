"""Predeclare data-size and optimizer-step controls before formal runs."""
from pathlib import Path
import subprocess,sys,json,platform,hashlib
from datetime import datetime,timezone
import torch,transformers
root=Path('results_data');root.mkdir(exist_ok=True)
plan=[dict(percent=p,budget='epochs',mode=m,seeds=[17,42,73]) for p in (25,50) for m in ('full','bitfit')]+[dict(percent=25,budget='steps',mode=m,seeds=[17,42,73]) for m in ('full','bitfit')]
env={'declared_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),'training_script_sha256':hashlib.sha256(Path('train_data.py').read_bytes()).hexdigest(),'base_training_script_sha256':hashlib.sha256(Path('train_local.py').read_bytes()).hexdigest()}
(root/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
(root/'plan.json').write_text(json.dumps({'matrix':plan,'prior_full_data_controls':'results_local/{17,42,73}/{full_lr2e-05,bitfit_lr1e-03}','subset_sampling_seed':20261117,'subset_policy':'fixed stratified nested subsets, identical across training seeds and methods','epochs_budget':3,'matched_optimizer_steps':1377,'fixed_learning_rates':{'full':2e-5,'bitfit':1e-3},'selection':'no tuning, final iterate only; already-observed validation remains exploratory','gpu_input_cache':'full train+validation cache resident for all sizes, CPU index mapping selects training rows'},indent=2),encoding='utf8')
for cfg in plan:
 print('CONFIG',cfg,flush=True)
 subprocess.run([sys.executable,'-s','-u','train_data.py','--mode',cfg['mode'],'--percent',str(cfg['percent']),'--budget',cfg['budget']],check=True)
print('ALL 18 DATA EXPERIMENTS COMPLETED',flush=True)
