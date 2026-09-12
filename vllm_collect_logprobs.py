#!/usr/bin/env python3
"""Tokenize probe_text exactly like logits_probe.py, ask the running vLLM for prompt logprobs, save ids+logprobs."""
import sys, json, os, math, urllib.request
from transformers import AutoTokenizer
path, out = sys.argv[1], sys.argv[2]; base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8001"
tok = AutoTokenizer.from_pretrained(path)
TEXT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_text.txt")).read()
prompt = tok.apply_chat_template([{"role": "user", "content": TEXT}], tokenize=False, add_generation_prompt=True)
ids = tok(prompt).input_ids[:1024]
body = {"model": "agnes-3.0-flash", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=600).read())
plp = r["choices"][0]["prompt_logprobs"]; lps = []
for i, d in enumerate(plp):
    if d is None: continue
    t = str(ids[i]); lps.append(d[t]["logprob"] if t in d else max(v["logprob"] for v in d.values()))
json.dump({"ids": ids, "logprobs": lps}, open(out, "w"))
print(f"tokens {len(ids)} logprobs {len(lps)} served ppl {math.exp(-sum(lps)/len(lps)):.3f}")
