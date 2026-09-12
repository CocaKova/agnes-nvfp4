#!/usr/bin/env python3
"""Teacher-forced ppl on raw (untemplated) texts via /v1/completions prompt_logprobs. usage: raw_probe.py <label> [base]"""
import sys, json, math, urllib.request
label = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8001"
TEXTS = {
 "count": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30",
 "license": "Licensed under the Apache License, Version 2.0 (the \"License\"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
 "prose": "The quick brown fox jumps over the lazy dog. Paris is the capital of France, and Berlin is the capital of Germany. Water boils at 100 degrees Celsius at sea level. The Python programming language was created by Guido van Rossum and first released in 1991.",
}
for name, text in TEXTS.items():
    body = {"model": "agnes-3.0-flash", "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    plp = [d for d in r["choices"][0]["prompt_logprobs"] if d is not None]
    lps = [max(v["logprob"] for v in d.values()) if len(d)==1 else list(d.values())[0]["logprob"] for d in plp]
    # with prompt_logprobs=0 vLLM returns only the actual token's entry
    lps = [list(d.values())[0]["logprob"] for d in plp]
    print(f"{label:6s} {name:8s} tokens {len(lps):3d} ppl {math.exp(-sum(lps)/len(lps)):8.3f}  worst {min(lps):7.2f}")
