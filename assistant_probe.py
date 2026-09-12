#!/usr/bin/env python3
"""Quality probe on ASSISTANT tokens (the only positions SFT trains on).
  gen   <model_dir> <out.json>  : greedy-generate answers for fixed prompts on the running server, save token ids
  score <model_dir> <in.json> <label> : teacher-force those ids on the running server, ppl over assistant span only
"""
import sys, json, math, urllib.request
from transformers import AutoTokenizer
mode, path = sys.argv[1], sys.argv[2]; B = "http://127.0.0.1:8001"
tok = AutoTokenizer.from_pretrained(path)
PROMPTS = ["How many primes are there below 200? Answer with the number and a one-line justification.",
           "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two asserts.",
           "Explain in one paragraph why unified CPU/GPU memory changes how you size the KV cache when serving an LLM."]
def post(ep, body):
    req = urllib.request.Request(B + ep, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())
if mode == "gen":
    out = []
    for p in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, reasoning_effort="low")
        pids = tok(prompt).input_ids
        r = post("/v1/completions", {"model": "agnes-3.0-flash", "prompt": pids, "max_tokens": 700, "temperature": 0})
        text = r["choices"][0]["text"]; aids = tok(text).input_ids
        out.append({"prompt": p, "prompt_ids": pids, "answer_ids": aids, "text": text})
        print(f"gen: {len(pids)} prompt tok, {len(aids)} answer tok, finish {r['choices'][0]['finish_reason']} | {text[-100:]!r}")
    json.dump(out, open(sys.argv[3], "w"))
else:
    data = json.load(open(sys.argv[3])); label = sys.argv[4]; tot, n, worst = 0.0, 0, 0.0; per = []
    for d in data:
        ids = d["prompt_ids"] + d["answer_ids"]
        r = post("/v1/completions", {"model": "agnes-3.0-flash", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
        plp = r["choices"][0]["prompt_logprobs"]
        lps = [list(plp[i].values())[0]["logprob"] for i in range(len(d["prompt_ids"]), len(ids))]
        per.append(lps); tot += sum(lps); n += len(lps); worst = min(worst, min(lps))
    json.dump(per, open(sys.argv[3].replace(".json", f".{label}.scores.json"), "w"))
    print(f"{label}: assistant-token ppl {math.exp(-tot/n):.4f} over {n} tokens, worst token logprob {worst:.2f}")
