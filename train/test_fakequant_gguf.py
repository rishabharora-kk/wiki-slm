"""Bit-match check: fakequant.q4_0 vs gguf-py's reference Q4_0 quantize/dequantize (pip install gguf)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakequant import q4_0
from gguf.quants import quantize, dequantize
from gguf import GGMLQuantizationType as Q
torch.manual_seed(0)
for scale in [0.02, 1.0, 5.0]:
    w = torch.randn(256, 1536) * scale; w[0, :40] = 0
    ref = dequantize(quantize(w.numpy().astype(np.float32), Q.Q4_0), Q.Q4_0)
    d = np.abs(q4_0(w).numpy() - ref).max()
    print(scale, "max abs diff", d); assert d == 0, d
print("bit-exact vs gguf-py Q4_0")
