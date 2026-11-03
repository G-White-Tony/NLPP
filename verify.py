"""Recompute all reported metrics and independently check parameter-update evidence."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def verify():
    val = read(next(Path(".cache").glob("validation-*.json")))
    expected_ids = read(Path("data/validation_ids.json"))
    assert [r["idx"] for r in val] == expected_ids
    summary = []
    counts = {"full":109483778,"head":1538,"bitfit":104450,"query":10754}
    for seed in (17,42,73):
        heads = set()
        for mode, count in counts.items():
            path = Path("results") / str(seed) / mode
            m = read(path/"metrics.json")
            heads.add(m["initial_head_sha256"])
            assert m["steps"] == 1377 and m["train_examples"] == 3668
            preds = [json.loads(s) for s in (path/"predictions.jsonl").read_text().splitlines()]
            assert [r["idx"] for r in preds] == expected_ids
            assert [r["label"] for r in preds] == [r["label"] for r in val]
            cm = Counter((r["label"],r["prediction"]) for r in preds)
            tn, fp, fn, tp = (cm[k] for k in ((0,0),(0,1),(1,0),(1,1)))
            assert math.isclose(m["accuracy"], (tp+tn)/len(val))
            assert math.isclose(m["f1"], 2*tp/(2*tp+fn+fp))
            audit = read(path/"audit.json")
            assert len(audit) == len({a["name"] for a in audit}) == 201
            changed = [a for a in audit if a["changed_elements"]]
            assert sum(a["numel"] for a in changed) == count == m["trainable"]
            for a in audit:
                n = a["name"]
                allowed = mode=="full" or n.startswith("classifier.") or (mode=="bitfit" and n.endswith(".bias")) or (mode=="query" and n.endswith("query.bias"))
                assert allowed == a["trainable"] == (a["max_abs_delta"] > 0)
                if allowed:
                    assert math.isfinite(a["first_grad_norm"]) and a["first_grad_norm"] > 0
                else:
                    assert a["first_grad_norm"] is None
            summary.append(dict(seed=seed,mode=mode,F1=m["f1"],accuracy=m["accuracy"],trainable=count,
                                updated_tensors=len(changed),changed_elements=sum(a["changed_elements"] for a in changed),
                                seconds=m["seconds"],peak_MiB=m["peak_allocated_MiB"],TN=tn,FP=fp,FN=fn,TP=tp))
        assert len(heads) == 1
    with Path("results/summary.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    for mode in counts:
        subset = [r for r in summary if r["mode"] == mode]
        for k in ("F1","accuracy","seconds","peak_MiB"):
            values = [r[k] for r in subset]
            print(mode,k,mean(values),stdev(values))
    return summary

if __name__ == "__main__":
    verify()
