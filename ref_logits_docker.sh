#!/bin/bash
# Reference logits from the ORIGINAL Agnes checkpoint via its own modeling code (needs transformers>=5.12,
# which the local quant-env lacks) -> run inside the vLLM image that ships transformers 5.15.
set -eo pipefail
OUT=$HOME/.local/state/agnes/logits-agnes-orig.pt
docker run --rm --ipc=host -e PYTHONPATH=$HOME/.local/state/agnes/tf512 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $HOME/models:$HOME/models -v "$(cd "$(dirname "$0")" && pwd)":/w -v $HOME/.local/state/agnes:$HOME/.local/state/agnes \
  --entrypoint python3 vllm/vllm-openai:qwen38-flash-next /w/logits_probe.py agnes $HOME/models/agnes-3.0-flash-bf16 "$OUT"
