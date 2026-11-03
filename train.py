import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, set_seed

CACHE = Path(".cache")
MODEL = CACHE / "bert"
MODES = ("full", "head", "bitfit", "query")
SEEDS = (17, 42, 73)
BATCH = 8

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")

def dataset(split, tokenizer):
    rows = json.loads(next(CACHE.glob(split + "-*.json")).read_text(encoding="utf8"))
    assert len(rows) == {"train":3668, "validation":408}[split]
    ids = [r["idx"] for r in rows]
    assert len(ids) == len(set(ids))
    manifest = Path("data") / (split + "_ids.json")
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf8")) == ids, split
    else:
        write(manifest, ids)
    features = tokenizer([r["sentence1"] for r in rows], [r["sentence2"] for r in rows],
                         max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    features["labels"] = torch.tensor([r["label"] for r in rows])
    return rows, {k:v.cuda() for k,v in features.items()}

def run(mode, seed, train, validation, smoke):
    dest = Path("results") / ("smoke" if smoke else str(seed)) / mode
    dest.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model, info = BertForSequenceClassification.from_pretrained(MODEL, output_loading_info=True)
    assert set(info["missing_keys"]) == {"classifier.bias","classifier.weight"}, info
    initial = {n:p.detach().clone() for n,p in model.named_parameters()}
    head_hash = hashlib.sha256(initial["classifier.weight"].numpy().tobytes()).hexdigest()
    model.cuda()
    for n,p in model.named_parameters():
        enabled = mode == "full" or n.startswith("classifier.") or (mode == "bitfit" and n.endswith(".bias")) or (mode == "query" and n.endswith(".attention.self.query.bias"))
        p.requires_grad_(enabled)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = 2e-5 if mode == "full" else 1e-3
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    rows, features = train
    size = 16 if smoke else len(rows)
    epochs = 1 if smoke else 3
    steps = epochs * math.ceil(size/BATCH)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0, 1-step/steps))
    grads = {}
    losses = []
    generator = torch.Generator().manual_seed(seed)
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(epochs):
        total_loss = torch.zeros((), device="cuda")
        order = torch.randperm(size, generator=generator).cuda()
        for step, ix in enumerate(order.split(BATCH)):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**{k:v[ix] for k,v in features.items()}).loss
            loss.backward()
            if epoch == step == 0:
                grads = {n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            total_loss += loss.detach() * len(ix)
        losses.append(float(total_loss/size))
        assert math.isfinite(losses[-1])
        print(seed, mode, epoch+1, losses[-1], flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()/2**20
    audit = []
    for n,p in model.named_parameters():
        delta = (p.detach().cpu()-initial[n]).abs()
        changed = int(torch.count_nonzero(delta))
        audit.append(dict(name=n, trainable=p.requires_grad, numel=p.numel(), first_grad_norm=grads[n],
                          changed_elements=changed, max_abs_delta=float(delta.max())))
        if not p.requires_grad:
            assert changed == 0 and grads[n] is None
    assert all(a["changed_elements"] and a["first_grad_norm"] is not None for a in audit if a["trainable"])
    write(dest/"audit.json", audit)
    valrows, val = validation
    predictions = []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, 16 if smoke else len(valrows), BATCH):
            probs = model(**{k:v[i:i+BATCH] for k,v in val.items() if k!="labels"}).logits.float().softmax(-1).cpu()
            for j, prob in enumerate(probs):
                r = valrows[i+j]
                predictions.append(dict(idx=r["idx"], label=r["label"], prediction=int(prob.argmax()), p1=float(prob[1])))
    (dest/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in predictions), encoding="utf8")
    acc = sum(r["label"]==r["prediction"] for r in predictions)/len(predictions)
    tp = sum(r["label"]==r["prediction"]==1 for r in predictions)
    f1 = 2*tp/max(1,sum(r["label"]+r["prediction"] for r in predictions))
    write(dest/"metrics.json", dict(mode=mode,seed=seed,accuracy=acc,f1=f1,epochs=epochs,steps=steps,
          train_examples=size,validation_examples=len(predictions),lr=lr,batch=BATCH,losses=losses,
          seconds=seconds,peak_allocated_MiB=peak,trainable=sum(p.numel() for p in params),
          total=sum(p.numel() for p in model.parameters()),initial_head_sha256=head_hash))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--seed",type=int, choices=SEEDS)
    parser.add_argument("--mode",choices=MODES)
    args = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained(MODEL)
    train = dataset("train",tokenizer)
    validation = dataset("validation",tokenizer)
    for seed in ([args.seed] if args.seed else SEEDS[:1] if args.smoke else SEEDS):
        for mode in ([args.mode] if args.mode else MODES):
            run(mode, seed, train, validation, args.smoke)
