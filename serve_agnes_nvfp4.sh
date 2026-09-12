#!/bin/bash
# Serve the NVFP4 build on one DGX Spark with the local vLLM install (0.20.2rc1 tree, sm_121a).
# Same flags that served qwen3.8-27b-nvfp4-mtp here (compressed-tensors W4A4 + MTP + flashinfer).
set -eo pipefail
MODEL="${MODEL:-/home/cocakova/models/agnes-3.0-flash-nvfp4}"
PORT="${PORT:-8001}"
UTIL="${UTIL:-0.55}"
MAXLEN="${MAXLEN:-131072}"
SPEC_N="${SPEC_N:-3}"
LOG="${LOG:-/home/cocakova/.local/state/agnes/serve.log}"
source /home/cocakova/vllm-install/vllm_env.sh
export ENABLE_NVFP4_SM100=0 VLLM_USE_DEEP_GEMM=0
SPEC_ARG=""; [ "$SPEC_N" != "0" ] && SPEC_ARG="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_N}"
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --served-model-name agnes-3.0-flash \
    --gpu-memory-utilization "$UTIL" --max-num-seqs 4 \
    --max-model-len "$MAXLEN" --max-num-batched-tokens 8192 \
    --kv-cache-dtype bfloat16 --attention-backend flashinfer \
    $SPEC_ARG --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":4}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser deepseek_r1 \
    --chat-template "$MODEL/chat_template.jinja" \
    --no-enable-log-requests "$@" > "$LOG" 2>&1
