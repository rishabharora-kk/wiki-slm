"""Phase 0 on Kaggle T4x2: measure throughput, LR sweep (80M tokens each), resume check."""
import glob, os, subprocess, sys, json
def sh(c):
    print("$", c, flush=True); r = subprocess.run(c, shell=True); print("exit", r.returncode, flush=True); return r.returncode
sh("nvidia-smi -L; python -c 'import torch;print(torch.__version__, torch.cuda.device_count())'")
sh("git clone --depth 1 https://github.com/rishabharora-kk/wiki-slm /kaggle/working/repo")
meta = glob.glob("/kaggle/input/**/meta.json", recursive=True)
print("meta:", meta); D = os.path.dirname(meta[0])
print(open(meta[0]).read())
T = "torchrun --nproc_per_node=2 /kaggle/working/repo/train/train.py --data %s" % D
# 1) throughput + memory: 60 steps
sh(f"{T} --out /kaggle/working/tp --max_steps 60 --val_every 1000 --micro 16 2>&1 | tail -15")
# 2) LR sweep, 80M tokens each (~305 steps @262k tok/step)
for lr in ["1e-3", "2e-3", "4e-3"]:
    sh(f"{T} --out /kaggle/working/lr{lr} --lr {lr} --max_tokens 80e6 --milestones 1e12 --val_every 50 --micro 16 --warmup 100 2>&1 | grep -E 'VAL|tok/s|params|Error|error' | tail -12")
# 3) resume check: run 100 steps wall-killed, then resume to 150
sh(f"{T} --out /kaggle/working/res --max_steps 100 --ckpt_every_min 0 --val_every 50 2>&1 | tail -3")
sh(f"{T} --out /kaggle/working/res --max_steps 150 --val_every 50 2>&1 | grep -E 'resumed|VAL' | tail -4")
summary = {}
for d in glob.glob("/kaggle/working/lr*"):
    rows = [json.loads(l) for l in open(d + "/val.jsonl")] if os.path.exists(d + "/val.jsonl") else []
    summary[os.path.basename(d)] = rows[-1] if rows else None
json.dump(summary, open("/kaggle/working/phase0_summary.json", "w"), indent=1); print(summary)
sh("rm -rf /kaggle/working/*/ckpt_* ")  # keep outputs small
