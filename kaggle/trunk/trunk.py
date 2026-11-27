"""Trunk session: resumes from previous session's output (if any), trains for WALL minutes, saves ckpts."""
import glob, os, shutil, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
WALL = int(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wall.txt")).read())
def sh(c): print("$", c, flush=True); return subprocess.run(c, shell=True).returncode
sh("git clone --depth 1 https://github.com/rishabharora-kk/wiki-slm /kaggle/working/repo")
D = os.path.dirname(glob.glob("/kaggle/input/**/meta.json", recursive=True)[0])
OUT = "/kaggle/working/trunk"; os.makedirs(OUT, exist_ok=True)
for f in glob.glob("/kaggle/input/**/trunk/ckpt_*.pt", recursive=True) + glob.glob("/kaggle/input/**/trunk/val.jsonl", recursive=True):
    print("carry", f); shutil.copy(f, OUT)
sh(f"torchrun --nproc_per_node=2 /kaggle/working/repo/train/train.py --data {D} --out {OUT} --lr 2e-3 --micro 8 --compile "
   f"--max_wall_min {WALL} --ckpt_every_min 20 --val_every 250 2>&1 | grep -vE 'Warning|warn' | awk 'NR%1==0' | tail -400")
sh(f"ls -la {OUT}; tail -3 {OUT}/val.jsonl")
