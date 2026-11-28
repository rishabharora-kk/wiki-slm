"""Branch session: cooldown arms from trunk milestones. C0 = plain cooldown (+EMA), C2 = Q4_0 QAT cooldown."""
import glob, os, shutil, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
JOBS = [("2.0B", 2e9, "C0"), ("2.0B", 2e9, "C2")]  # (ckpt tag, milestone tokens, arm)
SEED = 0
def sh(c): print("$", c, flush=True); return subprocess.run(c, shell=True).returncode
sh("git clone --depth 1 https://github.com/rishabharora-kk/wiki-slm /kaggle/working/repo")
D = os.path.dirname(glob.glob("/kaggle/input/**/meta.json", recursive=True)[0])
CK = {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/trunk/ckpt_*B.pt", recursive=True)}
print(CK)
for tag, M, arm in JOBS:
    out = f"/kaggle/working/br/{tag}_{arm}_s{SEED}"; os.makedirs(out, exist_ok=True)
    base = (f"torchrun --nproc_per_node=2 /kaggle/working/repo/train/train.py --data {D} --out {out} --lr 2e-3 --micro 8 "
            f"--branch_from {CK['ckpt_%s.pt' % tag]} --max_tokens {M:g} --cooldown_frac 0.2 --val_every 100 --seed {SEED} "
            + ("--ema 0.998 " if arm == "C0" else "--qat "))
    rc = sh(base + "--compile 2>&1 | grep -E 'step|VAL|rror|branched|resumed' | tail -60")
    if not os.path.exists(f"{out}/final.pt"):
        print("compile run produced no final.pt; retrying without --compile", flush=True)
        sh(base + "2>&1 | grep -E 'step|VAL|rror|branched|resumed' | tail -60")
    for f in glob.glob(f"{out}/ckpt_*"): os.remove(f)
    sh(f"ls -la {out}; tail -2 {out}/val.jsonl")
