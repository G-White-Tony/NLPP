# Local experiment additions

This directory preserves the source archive results in `results/`. New experiments write only to `results_local/` and do not replace those archived records.

## Scope fixed before formal local runs

- BERT-base-uncased and MRPC revisions and SHA-256 values in `data/resources.json`.
- 3668 training examples, 408 validation examples, seeds 17 / 42 / 73.
- Three epochs, batch 8, 1377 steps, max length 128; BF16 autocast and FP32 parameters.
- Attention implementation is explicitly `eager` to match the original pre-SDPA implementation.
- Optimizer / schedule / dropout / seed and data order follow the archived training code.
- No validation-based checkpoint selection. Always evaluate epoch 3. No test-set use.
- Seven configurations (21 formal runs): `full` 2e-5; `head` 1e-3 / 1e-4 / 4e-4; `bitfit`, `query`, `query_intermediate` each 1e-3.
- `query_intermediate` allows query.bias and intermediate.dense.bias in all 12 layers plus classifier; pooler remains frozen, as in `query`.
- Learning-rate comparisons are exploratory on the already-observed validation set. All settings, not just the best, are reported.

## Environment installation

In a separate Python 3.13 environment, install the observed training versions (do not upgrade an unrelated existing environment):

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.46.0 numpy==2.3.3 tokenizers==0.20.3 safetensors==0.7.0 pyarrow==24.0.0
```

`requirements-local.txt` records the installed versions used in the local execution. The original README is preserved separately in the delivery archive as README_ARCHIVE.md and describes the older environment.

## Commands

Use the environment versions recorded by `results_local/environment.json`. Local execution used `D:/miniconda3/python.exe`, without altering the base environment.

```powershell
python -s prepare.py
python -s train_local.py --smoke
python -s -u run_suite.py
python -s verify_local.py
```

One run: `python -s train_local.py --mode head --seed 17 --lr 1e-4`.
Completed runs are skipped if their metrics.json exists; remove a run directory only if intentionally rerunning it. Results are grouped by seed/configuration so rates do not overwrite each other.

## Evidence

- `results_local/plan.json`: declared matrix, shared budget, reporting policy.
- `results_local/environment.json`: actual Python / PyTorch / Transformers / GPU driver and training-script hash.
- `results_local/suite.log`: training progress with UTC timestamps.
- `results_local/<seed>/<config>/metrics.json`: configuration, losses, metrics, actual timings and GPU memory.
- `predictions.jsonl`: ordered validation ids, original labels, hard predictions, positive probabilities.
- `audit.json`: all 201 parameter tensors, initial gradient norms, changed element counts and maximum differences.
- `verify_local.py`: source-file hash checks, labels / metrics recomputation, allowed parameter checks and initialization checks; produces summary_local.csv, aggregate.json and verification.json.

No model checkpoint is included in the evidence ZIP. `prepare.py` downloads the exact pinned weights and verifies them. Resource timings are training-only, including first-step gradient auditing, excluding validation, downloads, disk writes and post-training parameter-difference auditing. Peak memory is PyTorch allocated MiB, not total GPU occupancy. All tokenized training and validation inputs remain resident on the GPU. Background applications and thermal conditions can affect time; runs are sequential, not simultaneous. No speed comparison between the archived and local software environments is claimed.
