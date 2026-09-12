#!/usr/bin/env python3
"""Zero-pad the MTP head's SwiGLU to the folded intermediate_size.
Agnes's MTP layer has no parallel_ffn branch (17408 wide) but the folded main graph is 19456 wide, and
vLLM sizes the MTP MLP from text_config.intermediate_size. Zero rows in gate/up and zero columns in
down contribute exactly 0 (silu(0)*0 = 0), so this is exact.
usage: pad_mtp_mlp.py <dir with model-mtp-bf16.safetensors | folded bf16 dir> <target_intermediate>"""
import sys, os, json, torch
from safetensors import safe_open
from safetensors.torch import save_file
d, target = sys.argv[1], int(sys.argv[2])
idx_p = os.path.join(d, "model.safetensors.index.json"); wm = json.load(open(idx_p))["weight_map"]
names = [k for k in wm if k.startswith("mtp.layers.0.mlp.")]
files = sorted({wm[k] for k in names}); assert len(files) == 1, files
f = files[0]; path = os.path.join(d, f)
with safe_open(path, framework="pt") as sf:
    tensors = {k: sf.get_tensor(k) for k in sf.keys()}
for k in names:
    t = tensors[k]
    if k.endswith("down_proj.weight"):
        assert t.shape[1] <= target; pad = torch.zeros(t.shape[0], target - t.shape[1], dtype=t.dtype)
        tensors[k] = torch.cat([t, pad], dim=1).contiguous()
    else:
        assert t.shape[0] <= target; pad = torch.zeros(target - t.shape[0], t.shape[1], dtype=t.dtype)
        tensors[k] = torch.cat([t, pad], dim=0).contiguous()
    print(k, tuple(t.shape), "->", tuple(tensors[k].shape))
os.replace(path, path + ".bak"); save_file(tensors, path, metadata={"format": "pt"}); os.remove(path + ".bak")
idx = json.load(open(idx_p)); idx["metadata"]["total_size"] = sum(os.path.getsize(os.path.join(d, x)) for x in set(wm.values()))
json.dump(idx, open(idx_p, "w"), indent=2); print("rewrote", f)
