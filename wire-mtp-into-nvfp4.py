#!/usr/bin/env python3
"""Wire the bf16 MTP draft head into an NVFP4 build, mirroring the layout of
models/qwen3.8-27b-nvfp4-mtp (the known-good reference for this arch).

Why this exists: transformers' Qwen3_5ForConditionalGeneration has no MTP
submodule, so quantization -- however it is invoked -- always drops the 15
mtp.* tensors. They have to be lifted out of the bf16 source afterwards and
added to the index by hand, and their module names added to
quantization_config.ignore.

The ignore entries are load-bearing: if the server treats the bf16 draft head
as NVFP4 the draft breaks silently -- 0% acceptance and slower than no MTP.

Usage: wire-mtp-into-nvfp4.py <bf16_source_dir> <nvfp4_dir>
"""
import json, os, struct, sys
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1], sys.argv[2]

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h

# ---- 1. pull the mtp.* tensors out of the bf16 source -----------------------
src_idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
mtp_names = sorted(k for k in src_idx if k.startswith("mtp"))
if not mtp_names:
    sys.exit("ABORT: no mtp.* tensors in the bf16 source")
print(f"[1/4] lifting {len(mtp_names)} mtp tensors from {SRC}")

tensors = {}
for name in mtp_names:
    shard = os.path.join(SRC, src_idx[name])
    with safe_open(shard, framework="pt") as f:
        t = f.get_tensor(name)
    assert str(t.dtype) == "torch.bfloat16", f"{name} is {t.dtype}, expected bfloat16"
    tensors[name] = t

mtp_path = os.path.join(DST, "model-mtp-bf16.safetensors")
save_file(tensors, mtp_path, metadata={"format": "pt"})
print(f"[1/4] wrote {mtp_path} ({os.path.getsize(mtp_path)/1e6:.1f} MB)")

# ---- 2. rebuild the index over quantized shards + the mtp file --------------
print("[2/4] rebuilding index")
shards = sorted(f for f in os.listdir(DST)
                if f.endswith(".safetensors") and f != "model-mtp-bf16.safetensors")
weight_map, total = {}, 0
for s in shards:
    for k in header(os.path.join(DST, s)):
        weight_map[k] = s
    total += os.path.getsize(os.path.join(DST, s))
for k in mtp_names:
    weight_map[k] = "model-mtp-bf16.safetensors"
total += os.path.getsize(mtp_path)

json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)
print(f"[2/4] index: {len(weight_map)} tensors across {len(shards)+1} files")

# ---- 3. add mtp modules to quantization_config.ignore -----------------------
print("[3/4] patching quantization_config.ignore")
cfg_path = os.path.join(DST, "config.json")
cfg = json.load(open(cfg_path))
qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config")
if qc is None:
    sys.exit("ABORT: no quantization_config in the output config")
# module name = tensor name minus the trailing parameter (".weight")
modules = sorted({n.rsplit(".", 1)[0] for n in mtp_names})
before = len(qc.get("ignore", []))
qc["ignore"] = sorted(set(qc.get("ignore", [])) | set(modules))
json.dump(cfg, open(cfg_path, "w"), indent=2)
print(f"[3/4] ignore: {before} -> {len(qc['ignore'])} (+{len(modules)} mtp modules)")

# ---- 4. verify -------------------------------------------------------------
print("[4/4] verifying")
idx = json.load(open(os.path.join(DST, "model.safetensors.index.json")))["weight_map"]
nvis = sum(1 for k in idx if "visual" in k)
nmtp = sum(1 for k in idx if k.startswith("mtp"))
qc = json.load(open(cfg_path))
qc = qc.get("quantization_config") or qc["text_config"]["quantization_config"]
nign = sum(1 for k in qc["ignore"] if k.startswith("mtp"))
print(f"    vision tensors: {nvis}")
print(f"    mtp tensors:    {nmtp}")
print(f"    mtp in ignore:  {nign}")
if nvis == 0 or nmtp != 15 or nign != 15:
    sys.exit("ABORT: verification failed")
print("    OK")
