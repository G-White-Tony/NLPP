# Further experiment: training-data size and budget control

This extension leaves train_local.py and all earlier results_local files unchanged. New training uses train_data.py and writes results_data. The original archive results remain in results.

## Matrix fixed before results

- 25% (917 examples) and 50% (1834 examples) of MRPC training, with full and BitFit, three epochs, seeds 17/42/73: 12 runs.
- 25% training with full and BitFit, exactly 1377 optimizer steps, seeds 17/42/73: 6 runs.
- Existing 100% three-epoch full/BitFit runs from results_local are reused only after checking identical resource hashes, initialization hashes, software versions, learning rates and precision.
- Total new runs: 18. Total locally trained formal runs over both extensions: 39. Archived 12 runs remain separate.

## Sampling

make_subsets.py creates fixed stratified nested subsets using sampling seed 20261117. Class-specific shuffled lists are truncated using largest-remainder allocation to exact sizes, and selected rows retain original source order. A single sample per fraction is shared across all training seeds and both methods. This design measures training randomness conditional on those samples, not uncertainty over dataset resampling. data/subsets/train_25.json and train_50.json preserve labels counts, source positions and actual IDs.

## Shared controls and interpretation

Model/data revisions, SHA-256, batch 8, maximum length 128, BF16 autocast/FP32 parameters, eager attention, optimizer/schedule and fixed learning rates follow the preceding local experiment. Full uses 2e-5 and BitFit 1e-3; no retuning or validation checkpoint selection. Evaluate the final iterate only. The validation set has already been observed; comparisons remain exploratory, not blind tests.

Three epochs means 345 steps for 25%, 690 for 50%, 1377 for 100%. Thus this curve changes both unique data and optimization budget. The 25% matched-step experiment repeats its 917 unique examples for 11 full passes plus 112 final batches, reaching exactly 1377 steps and 10983 processed examples, compared with 11004 processed examples in full-data three-epoch runs. Equal steps is not equal unique data or precisely equal sample exposure.

All methods keep the full tokenized training and validation tensors resident on GPU, selecting subset rows via CPU index mapping. This controls cache allocation across sizes; the recorded peak GPU allocated MiB cannot be used as a low-data input-storage saving claim. Timing includes training and first-gradient auditing, excludes evaluation, downloads, writes and final delta auditing. New runs use no model checkpoint from previous runs.

## Reproduce

Use requirements-local.txt and the preceding local environment (Python3.13.5, torch2.6.0+cu124, Transformers4.46.0). No packages were changed for this extension.

```powershell
python -s prepare.py
python -s make_subsets.py
python -s train_data.py --mode full --percent 25 --smoke
python -s train_data.py --mode bitfit --percent 25 --smoke
python -s run_data_suite.py
python -s verify_local.py
python -s verify_data.py
```

One run: python -s train_data.py --mode bitfit --percent 25 --budget steps --seed 17.
The CPU-only verify_data.py independently checks resource hashes, source labels, subset membership, disjointness from validation, nestedness, class counts, optimizer steps/sample exposure, all parameter gradients/deltas, initialization, and environment consistency. New results include per-run metrics.json / predictions.jsonl / audit.json, summary_data.csv, aggregate.json, verification.json, plan.json, environment.json, and suite.log. Trained weights are not included; pinned pretrained weights are downloaded by prepare.py. New script changes relative to train_local.py are saved in data_changes.diff.
