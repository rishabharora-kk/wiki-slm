"""Tokenize English Wikipedia to uint16 shards. Run on Kaggle (CPU or GPU session, internet on).
Split: sha1(id) % 1000 -> 0-4 val, 5-9 test, rest train. Articles joined with EOS."""
import hashlib, json, os, sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

OUT = sys.argv[1] if len(sys.argv) > 1 else ("/kaggle/working/out" if os.path.isdir("/kaggle/working") else "out")
LIMIT = int(os.environ.get("LIMIT", 0))  # articles, for smoke tests
SHARD = 500_000_000
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
EOS = tok.eos_token_id
assert tok.vocab_size < 2**16
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
if LIMIT: ds = ds.select(range(LIMIT))

def bucket(i): return int(hashlib.sha1(str(i).encode()).hexdigest(), 16) % 1000
ids_seen = {"train": set(), "val": set(), "test": set()}
counts = {"train": 0, "val": 0, "test": 0}
buf, shard_i = [], 0
val, test = [], []

def flush(final=False):
    global buf, shard_i
    while len(buf) >= SHARD or (final and buf):
        arr = np.array(buf[:SHARD], dtype=np.uint16); buf = buf[SHARD:]
        arr.tofile(f"{OUT}/train_{shard_i:03d}.bin"); shard_i += 1

B = 2000
for s in range(0, len(ds), B):
    rows = ds[s:s + B]
    keep = [(i, t) for i, t in zip(rows["id"], rows["text"]) if len(t.strip()) >= 500]
    enc = tok([t.strip() for _, t in keep], add_special_tokens=False)["input_ids"]
    for (i, _), e in zip(keep, enc):
        b = bucket(i); split = "val" if b < 5 else "test" if b < 10 else "train"
        ids_seen[split].add(i); counts[split] += len(e) + 1
        (val if split == "val" else test if split == "test" else buf).extend(e + [EOS])
    flush()
    if (s // B) % 50 == 0: print(s, counts, flush=True)
flush(final=True)
np.array(val, dtype=np.uint16).tofile(f"{OUT}/val.bin")
np.array(test, dtype=np.uint16).tofile(f"{OUT}/test.bin")
assert not (ids_seen["train"] & ids_seen["val"]) and not (ids_seen["train"] & ids_seen["test"]) and not (ids_seen["val"] & ids_seen["test"])
json.dump({"tokens": counts, "shards": shard_i, "tokenizer": "HuggingFaceTB/SmolLM2-135M",
           "dataset": "wikimedia/wikipedia 20231101.en"}, open(f"{OUT}/meta.json", "w"), indent=1)
print("done", counts)
