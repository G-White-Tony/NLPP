"""Nine equal-parameter layer-placement runs; declared before formal execution."""
from pathlib import Path
import json,sys,subprocess,platform,hashlib
from datetime import datetime,timezone
import torch,transformers
root=Path('results_layers');root.mkdir(exist_ok=True)
plan={'declared_utc':datetime.now(timezone.utc).isoformat(),'layer_indexing':'zero-based','matrix':{'bottom':list(range(6)),'top':list(range(6,12)),'alternating':list(range(0,12,2))},'seeds':[17,42,73],'trainable_per_run':52226,'encoder_bias_trainable':50688,'classifier_trainable':1538,'frozen_common':['embedding parameters including LayerNorm bias','pooler weight and bias','all encoder non-bias parameters'],'shared':{'train_examples':3668,'validation_examples':408,'epochs':3,'steps':1377,'batch':8,'lr':1e-3,'precision':'bf16_autocast_fp32_parameters','attention':'eager'},'selection':'final epoch, no validation selection; all arms reported; exploratory validation comparison'}
(root/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
env={'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),'training_script_sha256':hashlib.sha256(Path('train_layers.py').read_bytes()).hexdigest()}
(root/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
for mode in plan['matrix']:
 print('CONFIG',mode,plan['matrix'][mode],flush=True)
 subprocess.run([sys.executable,'-s','-u','train_layers.py','--mode',mode],check=True)
print('ALL 9 LAYER EXPERIMENTS COMPLETED',flush=True)
