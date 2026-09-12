#!/bin/bash
# Agnes-3.0-Flash Preview (folded Qwen3.5 graph, bf16 ~62 GiB) -> NVFP4 (~20 GiB)
# Needs ~70 GiB free: stop the brain first (GLM dual is TP2 -> stop on the head).
set -eo pipefail
VENV=/home/cocakova/quant-env
INPUT=/home/cocakova/models/agnes-3.0-flash-qwen35-bf16
OUTPUT=/home/cocakova/models/agnes-3.0-flash-nvfp4
LOG=/home/cocakova/.local/state/agnes/quant.log
D="$(cd "$(dirname "$0")" && pwd)"
[ -f "$INPUT/model.safetensors.index.json" ] || { echo "no folded source"; exit 1; }
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
[ -f "$LOG" ] && mv "$LOG" "$LOG.prev"
python3 "$D/quant_agnes_nvfp4.py" 2>&1 | tee -a "$LOG"
for f in chat_template.jinja tokenizer.json tokenizer_config.json vocab.json merges.txt \
         preprocessor_config.json video_preprocessor_config.json generation_config.json LICENSE; do
    [ -f "$OUTPUT/$f" ] || cp -v "$INPUT/$f" "$OUTPUT/$f" 2>/dev/null || true
done
python3 "$D/wire-mtp-into-nvfp4.py" "$INPUT" "$OUTPUT" 2>&1 | tee -a "$LOG"
echo "=== Done -> $OUTPUT" | tee -a "$LOG"
