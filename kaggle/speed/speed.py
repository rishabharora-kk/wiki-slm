"""Speed test on T4x2: baseline vs torch.compile, micro 8 vs 12."""
import glob, os, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def sh(c): print("$", c, flush=True); subprocess.run(c, shell=True)
sh("git clone --depth 1 https://github.com/rishabharora-kk/wiki-slm /kaggle/working/repo")
D = os.path.dirname(glob.glob("/kaggle/input/**/meta.json", recursive=True)[0])
T = f"torchrun --nproc_per_node=2 /kaggle/working/repo/train/train.py --data {D} --max_steps 120 --val_every 100000 --ckpt_every_min 999"
for name, extra in [("base8", "--micro 8"), ("compile8", "--micro 8 --compile"), ("base12", "--micro 12 --global_batch 288")]:
    sh(f"{T} --out /kaggle/working/{name} {extra} 2>&1 | grep -E 'tok/s|rror|OutOfMemory' | tail -4; echo ==={name}")
