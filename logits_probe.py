#!/usr/bin/env python3
"""Teacher-forced logits on a fixed text, for comparing checkpoints.
usage: logits_probe.py {agnes|qwen35} <model_dir> <out.pt>
  agnes  -> loads via trust_remote_code (the Agnes modeling file)
  qwen35 -> loads via transformers' Qwen3_5ForConditionalGeneration
Saves float32 logits for every position + the input ids.
"""
import sys, os, torch, json
DEV = os.environ.get("PROBE_DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "20")))
mode, path, out = sys.argv[1:4]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path)
TEXT = open(__file__.replace("logits_probe.py", "probe_text.txt")).read()
msgs = [{"role": "user", "content": TEXT}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").input_ids[:, :1024]
print("tokens:", ids.shape[1], flush=True)
if mode == "agnes":
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True, device_map=DEV)
else:
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV)
model.eval()
with torch.no_grad():
    logits = model(input_ids=ids.to(DEV)).logits[0].float().cpu()
torch.save({"ids": ids[0], "logits": logits}, out)
lp = torch.log_softmax(logits[:-1], -1); nll = -lp.gather(1, ids[0, 1:, None]).mean().item()
print(f"saved {out}; teacher-forced ppl {torch.exp(torch.tensor(nll)).item():.3f}", flush=True)
