#!/usr/bin/env python3
"""Teacher-forced logprobs from a running vLLM server on the SAME token ids as logits_probe.py,
compared with a saved bf16 HF logits file. usage: vllm_ppl_probe.py <ref.pt> [http://127.0.0.1:8001]"""
import sys, json, math, torch, urllib.request
ref = torch.load(sys.argv[1]); base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8001"
ids = ref["ids"].tolist()
body = {"model": "agnes-3.0-flash", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0, "logprobs": 0}
req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=600).read())
plp = r["choices"][0]["prompt_logprobs"]  # list, first is None
served = []
for i, d in enumerate(plp):
    if d is None: continue
    tok = str(ids[i]); served.append(d[tok]["logprob"] if tok in d else max(v["logprob"] for v in d.values()))
lp = torch.log_softmax(ref["logits"][:-1], -1); refl = lp.gather(1, ref["ids"][1:, None])[:, 0].tolist()
n = min(len(served), len(refl)); s, f = served[:n], refl[:n]
print(f"positions {n}")
print(f"ppl served {math.exp(-sum(s)/n):.3f}   ppl bf16 {math.exp(-sum(f)/n):.3f}")
print(f"mean |dlogprob| {sum(abs(a-b) for a,b in zip(s,f))/n:.4f}   max {max(abs(a-b) for a,b in zip(s,f)):.3f}")
