"""Q4_0-exact fake quantization (ggml reference): blocks of 32 along the last dim,
d = signed max-abs / -8 stored as fp16, q = clamp(int(x/d + 8.5), 0, 15), deq = (q-8)*d."""
import torch


def q4_0(w: torch.Tensor) -> torch.Tensor:
    shp = w.shape
    x = w.float().reshape(-1, 32)
    mx = x.gather(1, x.abs().argmax(dim=1, keepdim=True))
    d = (mx / -8).half().float()
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = (x * inv + 8.5).trunc().clamp(0, 15)
    return ((q - 8) * d).reshape(shp).to(w.dtype)


def q4_0_ste(w):  # forward quantized, backward identity
    return w + (q4_0(w) - w).detach()


if __name__ == "__main__":
    w = torch.randn(64, 256)
    q = q4_0(w)
    assert q.shape == w.shape and (q4_0(q) - q).abs().max() < 1e-5  # idempotent
    assert len(torch.unique(q.reshape(-1, 32)[0])) <= 16
    print("fakequant ok")
