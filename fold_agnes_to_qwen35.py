#!/usr/bin/env python3
"""Rewrite Agnes-AI/Agnes-3.0-Flash (Preview) as a stock Qwen3_5ForConditionalGeneration checkpoint.

Agnes-3.0-Flash is Qwen3.5-27B's architecture with 72 layers and one extra piece:
a second narrow SwiGLU (parallel_ffn, 2048 wide) per layer whose output is ADDED
to the main MLP's output. That is exactly a wider SwiGLU:

    down(silu(gate x) * up x) + down2(silu(gate2 x) * up2 x)
  = [down | down2] @ concat(silu([gate;gate2] x) * ([up;up2] x))

so gate/up concatenate along the output dim, down along the input dim, and
intermediate_size becomes 17408 + 2048 = 19456. Exact up to bf16 summation order
(the model's own SGLang patch does the same fold at load time).

Renames: delta_attn.* -> linear_attn.*, global_attn.* -> self_attn.* (also in mtp.*).
Everything else (vision tower, MTP head, norms, embeddings) passes through untouched.
Streams tensor-by-tensor; peak RSS ~ FLUSH_GIB + one layer.
"""
import json, os, sys, re, shutil, struct, torch
from collections import OrderedDict
from safetensors import safe_open
from safetensors.torch import save_file

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/cocakova/models/agnes-3.0-flash-bf16"
DST = sys.argv[2] if len(sys.argv) > 2 else "/home/cocakova/models/agnes-3.0-flash-qwen35-bf16"
FLUSH_GIB = float(os.environ.get("FLUSH_GIB", "3"))
os.makedirs(DST, exist_ok=True)

idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
cfg = json.load(open(os.path.join(SRC, "config.json")))
tc = cfg["text_config"]
L = tc["num_hidden_layers"]; I = tc["intermediate_size"]; P = tc.get("parallel_ffn_intermediate_size", 0) or 0
print(f"layers={L} intermediate={I} parallel={P} tensors={len(idx)}", flush=True)

handles = {}
def get(name):
    shard = idx[name]
    if shard not in handles:
        handles[shard] = safe_open(os.path.join(SRC, shard), framework="pt")
    return handles[shard].get_tensor(name)

def rename(k):
    k = k.replace(".delta_attn.", ".linear_attn.").replace(".global_attn.", ".self_attn.")
    return k

# ---- plan the output order: non-layer tensors, then layer by layer ----------
layer_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
plain = sorted(k for k in idx if not layer_re.match(k))
out_names = []          # (out_name, source spec)
for k in plain:
    out_names.append((rename(k), ("copy", k)))
for n in range(L):
    pre = f"model.language_model.layers.{n}."
    keys = sorted(k for k in idx if k.startswith(pre))
    par = {k for k in keys if ".mlp.parallel_ffn." in k}
    for k in keys:
        if k in par:
            continue
        if P and k.endswith(".mlp.gate_proj.weight") or P and k.endswith(".mlp.up_proj.weight"):
            out_names.append((rename(k), ("cat0", k, k.replace(".mlp.", ".mlp.parallel_ffn."))))
        elif P and k.endswith(".mlp.down_proj.weight"):
            out_names.append((rename(k), ("cat1", k, k.replace(".mlp.", ".mlp.parallel_ffn."))))
        else:
            out_names.append((rename(k), ("copy", k)))
    assert len(par) in (0, 3), (n, par)

# ---- stream out --------------------------------------------------------------
shard_id, buf, buf_bytes, weight_map, total = 0, OrderedDict(), 0, {}, 0
def flush():
    global shard_id, buf, buf_bytes
    if not buf: return
    shard_id += 1
    fn = f"model-{shard_id:05d}.safetensors"
    save_file(buf, os.path.join(DST, fn), metadata={"format": "pt"})
    for k in buf: weight_map[k] = fn
    print(f"  wrote {fn} ({buf_bytes/2**30:.2f} GiB, {len(buf)} tensors)", flush=True)
    buf, buf_bytes = OrderedDict(), 0

nfold = 0
for out, spec in out_names:
    if spec[0] == "copy":
        t = get(spec[1])
    elif spec[0] == "cat0":
        a, b = get(spec[1]), get(spec[2]); assert a.dtype == b.dtype == torch.bfloat16
        t = torch.cat([a, b], dim=0).contiguous(); nfold += 1
    else:
        a, b = get(spec[1]), get(spec[2]); assert a.dtype == b.dtype == torch.bfloat16
        t = torch.cat([a, b], dim=1).contiguous(); nfold += 1
    buf[out] = t; nb = t.numel() * t.element_size(); buf_bytes += nb; total += nb
    if buf_bytes >= FLUSH_GIB * 2**30: flush()
flush()
# rename shards to N-of-M
M = shard_id
for i in range(1, M + 1):
    old, new = f"model-{i:05d}.safetensors", f"model-{i:05d}-of-{M:05d}.safetensors"
    os.rename(os.path.join(DST, old), os.path.join(DST, new))
    for k, v in weight_map.items():
        if v == old: weight_map[k] = new
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)
print(f"folded {nfold} MLP tensors (expect {3*L}); out tensors {len(weight_map)} (expect {len(idx)-3*L}); {total/2**30:.2f} GiB", flush=True)
assert nfold == 3 * L and len(weight_map) == len(idx) - 3 * L

# ---- config.json in Qwen3.5 terms --------------------------------------------
ntc = dict(tc)
for k in ("bos_token_id", "pad_token_id", "tie_word_embeddings", "output_gate_type",
          "parallel_ffn_intermediate_size", "partial_rotary_factor", "global_attention_interval"):
    ntc.pop(k, None)
ntc["model_type"] = "qwen3_5_text"
ntc["full_attention_interval"] = tc.get("global_attention_interval", 4)
ntc["intermediate_size"] = I + P
ntc["mlp_only_layers"] = []
ntc["layer_types"] = ["linear_attention" if t == "agnes_delta_attention" else "full_attention" for t in tc["layer_types"]]
assert set(ntc["layer_types"]) == {"linear_attention", "full_attention"}
ntc["rope_parameters"].setdefault("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25))
nvc = dict(cfg["vision_config"]); nvc["model_type"] = "qwen3_5"
ncfg = {k: v for k, v in cfg.items() if k not in ("auto_map", "text_config", "vision_config")}
ncfg.update({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
             "text_config": ntc, "vision_config": nvc, "transformers_version": "5.10.1"})
json.dump(ncfg, open(os.path.join(DST, "config.json"), "w"), indent=2)

# ---- side files ----------------------------------------------------------------
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "chat_template.jinja", "generation_config.json", "LICENSE"):
    shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))
pp = json.load(open(os.path.join(SRC, "preprocessor_config.json"))); pp.pop("auto_map", None)
pp.update({"processor_class": "Qwen3VLProcessor", "image_processor_type": "Qwen2VLImageProcessorFast"})
json.dump(pp, open(os.path.join(DST, "preprocessor_config.json"), "w"), indent=2)
vp = json.load(open(os.path.join(SRC, "video_preprocessor_config.json"))); vp.pop("auto_map", None)
vp.update({"processor_class": "Qwen3VLProcessor", "video_processor_type": "Qwen3VLVideoProcessor"})
json.dump(vp, open(os.path.join(DST, "video_preprocessor_config.json"), "w"), indent=2)
print("DONE ->", DST, flush=True)
