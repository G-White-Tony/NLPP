"""Fetch pinned inputs and verify them before converting MRPC to JSON."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen
from shutil import copyfileobj
import pyarrow.parquet as pq

cache = Path(".cache")
hub = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
model = "google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594"
data = "datasets/nyu-mll/glue/resolve/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/mrpc"
for item in json.loads(Path("data/resources.json").read_text(encoding="utf-8-sig")):
    path = cache / item["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        root = data if path.suffix == ".parquet" else model
        temp = path.with_suffix(".tmp")
        with urlopen(f"{hub}/{root}/{path.name}", timeout=120) as src, temp.open("wb") as dst:
            copyfileobj(src, dst)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Invalid download: {path}")
        temp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Invalid cached input: {path}")
    if path.suffix == ".parquet":
        rows = pq.read_table(path).to_pylist()
        path.with_suffix(".json").write_text(json.dumps(rows), encoding="utf8")
