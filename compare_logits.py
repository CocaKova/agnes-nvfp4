#!/usr/bin/env python3
import sys, torch
a, b = (torch.load(p) for p in sys.argv[1:3])
assert torch.equal(a["ids"], b["ids"]), "different token ids"
la, lb = a["logits"], b["logits"]
pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
kl = (pa.exp() * (pa - pb)).sum(-1)
print(f"positions {la.shape[0]}  vocab {la.shape[1]}")
print(f"KL(a||b) mean {kl.mean():.3e}  max {kl.max():.3e}")
print(f"argmax agreement {(la.argmax(-1)==lb.argmax(-1)).float().mean():.4f}")
print(f"max |dlogit| {(la-lb).abs().max():.4f}   mean |dlogit| {(la-lb).abs().mean():.5f}")
ids = a["ids"][1:]
for n, lp in (("a", pa), ("b", pb)):
    nll = -lp[:-1].gather(1, ids[:, None]).mean(); print(f"ppl {n}: {nll.exp():.3f}")
