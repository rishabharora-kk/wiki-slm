"""Llama-style LM trainer for 2xT4: fp16+GradScaler, DDP, WSD schedule, resume, EMA, Q4_0 QAT.
  torchrun --nproc_per_node=2 train/train.py --data DIR --out DIR [flags]
Trunk: constant LR (stable phase of WSD). Branch (--branch_from CKPT): 1-sqrt cooldown over
cooldown_frac * trunk steps, fresh optimizer with 100-step warmup (identical for every arm)."""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.distributed as dist, torch.nn as nn
import torch.nn.utils.parametrize as P
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakequant import q4_0_ste

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--out", required=True)
p.add_argument("--d", type=int, default=512); p.add_argument("--L", type=int, default=8)
p.add_argument("--ffn", type=int, default=1536); p.add_argument("--heads", type=int, default=8)
p.add_argument("--kv", type=int, default=4); p.add_argument("--seq", type=int, default=1024)
p.add_argument("--micro", type=int, default=16); p.add_argument("--global_batch", type=int, default=256)
p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--warmup", type=int, default=500)
p.add_argument("--max_tokens", type=float, default=4e9)  # trunk length
p.add_argument("--milestones", default="0.5e9,1e9,2e9,4e9")
p.add_argument("--max_wall_min", type=float, default=660); p.add_argument("--ckpt_every_min", type=float, default=30)
p.add_argument("--val_every", type=int, default=250); p.add_argument("--seed", type=int, default=0)
p.add_argument("--branch_from"); p.add_argument("--cooldown_frac", type=float, default=0.2)
p.add_argument("--final_lr_frac", type=float, default=0.0)
p.add_argument("--ema", type=float, default=0); p.add_argument("--qat", action="store_true")
p.add_argument("--max_steps", type=int, default=0)  # smoke tests
a = p.parse_args()
T_START = time.time()

ddp = "RANK" in os.environ
rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
if ddp: dist.init_process_group("nccl")
cuda = torch.cuda.is_available()
dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
if cuda: torch.cuda.set_device(dev)
os.makedirs(a.out, exist_ok=True)
def log(*x):
    if rank == 0: print(*x, flush=True)

shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in sorted(glob.glob(f"{a.data}/train_*.bin"))]
cum = np.cumsum([len(s) for s in shards]); total = int(cum[-1])
val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint16)[:2_000_000]

def batch(idx, bsz):  # deterministic in (seed, idx, rank): resume needs only the step
    g = np.random.default_rng([a.seed, idx, rank])
    xs = []
    for o in g.integers(0, total - a.seq - 1, size=bsz):
        i = int(np.searchsorted(cum, o, side="right")); lo = int(o - (cum[i - 1] if i else 0))
        c = shards[i][lo:lo + a.seq + 1]
        if len(c) < a.seq + 1: c = np.concatenate([c, shards[(i + 1) % len(shards)][:a.seq + 1 - len(c)]])
        xs.append(c.astype(np.int64))
    t = torch.from_numpy(np.stack(xs)).to(dev)
    return t[:, :-1], t[:, 1:]

cfg = LlamaConfig(vocab_size=49152, hidden_size=a.d, intermediate_size=a.ffn, num_hidden_layers=a.L,
                  num_attention_heads=a.heads, num_key_value_heads=a.kv, max_position_embeddings=a.seq,
                  tie_word_embeddings=True, attn_implementation="sdpa", rms_norm_eps=1e-5)
raw = LlamaForCausalLM(cfg).to(dev)
log(f"params {sum(x.numel() for x in raw.parameters())/1e6:.1f}M")

tok_per_step = a.global_batch * a.seq
accum = a.global_batch // (a.micro * world)
trunk_steps = int(a.max_tokens // tok_per_step)
cool_steps = int(a.cooldown_frac * trunk_steps) if a.branch_from else 0
end_step = trunk_steps + cool_steps
miles = set() if a.branch_from else {int(float(m) // tok_per_step) for m in a.milestones.split(",")}
latest = f"{a.out}/ckpt_latest.pt"
resume = os.path.exists(latest)
ck = torch.load(latest, map_location=dev) if resume else None
branch_start = 0

def add_qat():
    class Q(nn.Module):
        def forward(self, w): return q4_0_ste(w)
    for m in raw.modules():  # every 2-D linear weight; tied embedding/lm_head stays fp (llama.cpp keeps it Q6_K/Q8_0-ish)
        if isinstance(m, nn.Linear) and m is not raw.lm_head: P.register_parametrization(m, "weight", Q())

def plain_sd():  # state dict with parametrizations stripped, loadable into a plain Llama
    return {k.replace("parametrizations.weight.original", "weight"): v for k, v in raw.state_dict().items()
            if "parametrizations" not in k or k.endswith("original")}

if not resume and a.branch_from:
    c = torch.load(a.branch_from, map_location=dev); raw.load_state_dict(c["model"])
    branch_start = trunk_steps = c["step"]; end_step = trunk_steps + cool_steps
if a.qat: add_qat()
if resume:
    raw.load_state_dict(ck["model"]); trunk_steps, end_step, branch_start = ck["trunk"], ck["end"], ck["branch_start"]

opt = torch.optim.AdamW(raw.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=cuda)
scaler = torch.amp.GradScaler(enabled=cuda)
step = branch_start if (a.branch_from and not resume) else 0
ema = None
if a.ema: ema = {k: v.detach().clone().float() for k, v in raw.state_dict().items()}
if resume:
    opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"]); step = ck["step"]
    if a.ema and ck["ema"] is not None: ema = ck["ema"]
    log("resumed at", step)
net = DDP(raw, device_ids=[dev.index]) if ddp else raw

def lr_at(s):
    if a.branch_from:
        w = min(1.0, (s - branch_start + 1) / 100)
        prog = min(1.0, max(0.0, (s - trunk_steps) / cool_steps))  # 1-sqrt cooldown (Dremov et al.)
        return a.lr * w * (a.final_lr_frac + (1 - a.final_lr_frac) * (1 - prog ** 0.5) if s >= trunk_steps else 1.0)
    return a.lr * min(1.0, (s + 1) / a.warmup)

def save(path, s):
    if rank: return
    torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": s,
                "ema": ema, "trunk": trunk_steps, "end": end_step, "branch_start": branch_start, "args": vars(a)},
               path + ".tmp")
    os.replace(path + ".tmp", path)

@torch.no_grad()
def evaluate():
    raw.eval(); tot = n = 0
    bs = a.micro
    for i in range(0, min(len(val) - a.seq * bs - 1, 40 * bs * a.seq), a.seq * bs):
        t = torch.from_numpy(val[i:i + a.seq * bs + 1].astype(np.int64))
        x = t[:-1].view(bs, a.seq).to(dev); y = t[1:].view(bs, a.seq).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=cuda):
            tot += raw(x, labels=y).loss.item(); n += 1
    raw.train(); return tot / max(n, 1)

t0 = time.time(); last_ck = time.time(); net.train()
while step < end_step and not (a.max_steps and step >= a.max_steps):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    for k in range(accum):
        x, y = batch(step * accum + k, a.micro)
        sync = not (ddp and k < accum - 1)
        with (net.no_sync() if not sync else torch.enable_grad()):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=cuda):
                loss = net(x, labels=y).loss / accum
            scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
    if ema is not None:
        with torch.no_grad():
            for k_, v in raw.state_dict().items(): ema[k_].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
    if step % 50 == 0:
        log(f"step {step} loss {loss.item()*accum:.4f} lr {lr_at(step):.2e} scale {scaler.get_scale():.0f} "
            f"tok/s {50*tok_per_step/(time.time()-t0):.0f}"); t0 = time.time()
    if step % a.val_every == 0 and rank == 0:
        v = evaluate(); log(f"VAL step {step} loss {v:.4f}")
        with open(f"{a.out}/val.jsonl", "a") as f: f.write(json.dumps({"step": step, "tokens": step * tok_per_step, "val": v}) + "\n")
    if step in miles: save(f"{a.out}/ckpt_{step*tok_per_step/1e9:.1f}B.pt", step)
    if time.time() - last_ck > a.ckpt_every_min * 60: save(latest, step); last_ck = time.time()
    if time.time() - T_START > a.max_wall_min * 60: break
save(latest, step)
if a.branch_from and step >= end_step and rank == 0:
    torch.save({"model": plain_sd(), "step": step}, f"{a.out}/final.pt")
    if ema is not None:
        torch.save({"model": {k.replace("parametrizations.weight.original", "weight"): v.half() for k, v in ema.items()
                              if "parametrizations" not in k or k.endswith("original")}}, f"{a.out}/final_ema.pt")
if ddp: dist.destroy_process_group()
