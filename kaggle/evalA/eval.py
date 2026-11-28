"""GGUF quantization eval (CPU). Builds llama.cpp, exports models, quantizes (--pure), measures KL vs F32 + PPL."""
import glob, json, os, re, subprocess, sys
import numpy as np, torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
def sh(c, quiet=False):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if not quiet: print("$", c[:200], "->", r.returncode, flush=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")
MODELS = [("R_1.0B", "ckpt_1.0B.pt"), ("C0_1.0B", "br/1.0B_C0_s0/final.pt"), ("C2_1.0B", "br/1.0B_C2_s0/final.pt")]
QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]
CHUNKS = 64
W = "/kaggle/working"; T = "/kaggle/temp"; os.makedirs(T, exist_ok=True)
sh(f"git clone https://github.com/ggml-org/llama.cpp {W}/llama.cpp")
_, sha = sh(f"cd {W}/llama.cpp && git rev-parse HEAD", True); print("llama.cpp commit", sha.strip(), flush=True)
rc, out = sh(f"cd {W}/llama.cpp && cmake -B build -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 --target llama-quantize llama-perplexity", True)
print("build rc", rc, out[-500:], flush=True)
B = f"{W}/llama.cpp/build/bin"
D = os.path.dirname(glob.glob("/kaggle/input/**/meta.json", recursive=True)[0])
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
test = np.fromfile(f"{D}/test.bin", dtype=np.uint16, count=CHUNKS * 1024 + 2048).astype(np.int64)
open(f"{T}/test.txt", "w").write(tok.decode(test))
cfg = LlamaConfig(vocab_size=49152, hidden_size=512, intermediate_size=1536, num_hidden_layers=8, num_attention_heads=8,
                  num_key_value_heads=4, max_position_embeddings=1024, tie_word_embeddings=True, rms_norm_eps=1e-5)
def find(rel):
    pats = glob.glob(f"/kaggle/input/**/{rel}", recursive=True) or glob.glob(f"/kaggle/input/**/{os.path.basename(rel)}", recursive=True)
    return [p for p in pats if rel.split('/')[0] in p or '/' not in rel][0]
def types(path):
    import gguf
    r = gguf.GGUFReader(path); c = {}
    for t in r.tensors: c[t.tensor_type.name] = c.get(t.tensor_type.name, 0) + 1
    return c
def parse(txt):
    d = {}
    for k, pat in [("ppl", r"Mean\s+PPL\(Q\)\s*:\s*([\d.]+)"), ("ppl_base", r"Mean\s+PPL\(base\)\s*:\s*([\d.]+)"),
                   ("kld", r"Mean\s+KLD:\s*([\d.]+)"), ("dnll", r"Mean\s+ln\(PPL\(Q\)/PPL\(base\)\):\s*([-\d.]+)"),
                   ("same_top_p", r"Same top p:\s*([\d.]+)"), ("rms_dp", r"RMS Δp\s*:\s*([\d.]+)")]:
        m = re.search(pat, txt); d[k] = float(m.group(1)) if m else None
    return d
res = open(f"{W}/eval.jsonl", "w")
for name, rel in MODELS:
    ck = torch.load(find(rel), map_location="cpu")["model"]
    m = LlamaForCausalLM(cfg); m.load_state_dict(ck); m = m.float()
    hf = f"{T}/hf_{name}"; m.save_pretrained(hf); tok.save_pretrained(hf)
    f32 = f"{T}/{name}.f32.gguf"
    rc, o = sh(f"python {W}/llama.cpp/convert_hf_to_gguf.py {hf} --outfile {f32} --outtype f32"); print(o[-300:] if rc else "", flush=True)
    kld = f"{T}/{name}.kld"
    rc, o = sh(f"{B}/llama-perplexity -m {f32} -f {T}/test.txt -c 1024 --chunks {CHUNKS} -t 4 --kl-divergence-base {kld}")
    base = parse(o); print(name, "F32", base, o[-300:] if rc else "", flush=True)
    res.write(json.dumps({"model": name, "quant": "F32", "size": os.path.getsize(f32), **base}) + "\n"); res.flush()
    for q in QUANTS:
        qf = f"{T}/{name}.{q}.gguf"
        rc, o = sh(f"{B}/llama-quantize --pure {f32} {qf} {q}")
        if rc: print(o[-300:]); continue
        rc, o = sh(f"{B}/llama-perplexity -m {qf} -f {T}/test.txt -c 1024 --chunks {CHUNKS} -t 4 --kl-divergence-base {kld} --kl-divergence")
        r = {"model": name, "quant": q, "size": os.path.getsize(qf), "types": types(qf), **parse(o)}
        print(r, flush=True); res.write(json.dumps(r) + "\n"); res.flush()
        os.remove(qf)
    os.remove(kld); os.remove(f32)
