# Creative ablation: equal parameter count, different bias location

This extension changes only which six encoder layers have trainable bias. All three arms have exactly 50,688 encoder bias elements plus 1,538 classifier elements = 52,226 trainable parameters, in 50 trainable tensors. Embeddings and pooler are frozen in every arm; all encoder non-bias parameters remain frozen. This is not a random-parameter or full-BitFit comparison.

## Layer placement (zero-based)

- bottom: 0,1,2,3,4,5 (first six layers nearest the input)
- top: 6,7,8,9,10,11 (last six layers nearest the output)
- alternating: 0,2,4,6,8,10 (one fixed interleaved placement, not an average over all placements)

## Shared budget

Full MRPC training 3668 / validation 408; same pinned model/data hashes; seeds 17,42,73; same initialized classifier and shuffled training sequence; batch8, max_length128, 3epochs/1377steps, lr1e-3, AdamW weight_decay0, linear decay to zero, no warmup, clipping1, BF16 autocast/FP32 parameters, eager attention. Always evaluate the final epoch; no validation selection. All current validation comparisons remain exploratory. Test set not used.

## Execute

Use the existing local environment and requirements-local.txt. New training uses train_layers.py, leaving train_local.py and train_data.py unchanged. results_layers/plan.json declares the arms and shared constraints before formal training; environment.json records software, GPU and code hash. Formal run count is 9; cumulative local count is 48 (prior 39 plus 9). Original archive 12 runs remain separate.

```powershell
python -s prepare.py
python -s train_layers.py --smoke
python -s run_layer_suite.py
python -s verify_layers.py
```

One run: python -s train_layers.py --mode top --seed 17.
Results are in results_layers/<seed>/<mode>_lr1e-03/ as metrics.json, audit.json and predictions.jsonl. summary_layers.csv and aggregate.json report all seeds and sample standard deviations; verification.json records independent checks of actual frozen-parameter deltas, first gradients, parameter count/layer membership, source labels/metrics, optimizer groups, initialized head and environment consistency. layer_changes.diff records the changes relative to train_local.py.

## Interpretation limits

Equal trainable parameter count controls parameter budget, not graph depth or representational capacity. Frozen layers still execute their forward pass. If trainable parameters appear only in the upper half, lower-layer computations need not retain the same backward graph as with trainable biases in the lower half. This is an autograd-based explanation, not a measured per-operator profile. Timing/peak allocated memory use the same policy as the earlier runs, with full tokenized input cache resident. Do not treat timings as platform-independent benchmarks. One interleaved placement cannot establish that all distributed placements are better/worse; three seeds on one task cannot establish universal linguistic layer roles.

## Single-batch graph check

`python -s inspect_layer_graph.py` records forward-hook observations in results_layers/graph_probe.json using seed17 and the first eight training rows. Top placement has no grad_fn for layer outputs 0--5, but has grad_fn for 6--11; bottom and alternating placements have grad_fn for all twelve outputs. One backward pass checks connectivity without an optimizer update. This check is separate from the nine formal training runs and their measured timings. It supports graph-coverage reasoning, not an operator-level decomposition of peak memory.
