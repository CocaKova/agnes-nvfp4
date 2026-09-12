---
license: apache-2.0
base_model: Agnes-AI/Agnes-3.0-Flash
base_model_relation: quantized
tags:
  - agnes
  - qwen3_5
  - nvfp4
  - compressed-tensors
  - vllm
  - blackwell
  - dgx-spark
  - multimodal
  - reasoning
library_name: transformers
pipeline_tag: image-text-to-text
quantized_by: CocaKova
---

# Agnes-3.0-Flash-Preview-NVFP4

NVFP4 (W4A4, group 16) build of [Agnes-AI/Agnes-3.0-Flash](https://huggingface.co/Agnes-AI/Agnes-3.0-Flash) (the open-weight **Preview** checkpoint, released 2026-09-11). Quantized on a single NVIDIA DGX Spark (GB10) with llm-compressor.

**61.6 GiB → 22.0 GiB.** Runs on one DGX Spark in stock vLLM with vision and the MTP draft head intact.

## What was done

1. **Folded to a stock Qwen3.5 graph.** Agnes-3.0-Flash is Qwen3.5-27B's architecture (gated delta rule 3:1 with gated full attention, mRoPE, same vision tower, one MTP layer) with 72 layers and one addition: a second, narrower SwiGLU per layer (`parallel_ffn`, 2048 wide) whose output is added to the main MLP. Adding two SwiGLUs on the same input is one wider SwiGLU, so gate/up were concatenated along the output dim, down along the input dim, and `intermediate_size` became 17408 + 2048 = 19456. The rename (`delta_attn→linear_attn`, `global_attn→self_attn`) is the same one the model's bundled SGLang patch applies at load time. Result: `Qwen3_5ForConditionalGeneration`, **no `trust_remote_code`**.
2. **Verified the fold** against the original modeling code: teacher-forced logits on the same 242 tokens, original checkpoint through its bundled `modeling_agnes.py` (transformers 5.12.1) vs the folded checkpoint through stock `Qwen3_5ForConditionalGeneration` (transformers 5.10.1): **mean KL 5.0e-3, argmax agreement 99.2%**, perplexity 164.3 vs 163.0. Same fold the bundled SGLang patch applies at load time.
3. **Quantized** with llm-compressor `NVFP4` (weights + activations FP4, per-group-16 FP8 scales), 32 × 4096-token calibration samples from ultrachat_200k. Kept in BF16: `lm_head`, the vision tower, the delta-rule `conv1d`, and the MTP head (`mtp.*`, re-attached after quantization and listed in `quantization_config.ignore`). The MTP layer has no parallel branch in the original, so its SwiGLU was zero-padded from 17408 to 19456 (exact) so vLLM's draft model matches the folded width.

## Quality (vLLM, one DGX Spark, same engine for both)

Teacher-forced perplexity. Assistant-token row = the bf16 model's own greedy answers to three prompts (2,001 assistant tokens), scored under each model — the only positions an SFT model is trained on.

| text | bf16 (folded, vLLM) | **NVFP4 (this build)** |
|---|---|---|
| assistant tokens (bf16 greedy answers, 2,001 tok) | 1.131 | **1.259** |
| raw prose (facts, 58 tok) | 2.836 | 2.984 |
| Apache license paragraph (87 tok) | 1.099 | 1.149 |
| counting 1..30 (108 tok) | 1.108 | 1.131 |

Greedy answers from the two models diverge early (as 4-bit builds do) but both reach the same conclusions (e.g. π(200) = 46, correct merge function with tests). Tool calling (`qwen3_xml` parser) and reasoning (`<think>`) verified end-to-end.

> Do not score system/user tokens: SFT never trains on them, both models are near-random there, and the numbers mean nothing (bf16 read 160 ppl on a user prompt, NVFP4 68 — and neither is "better").

## Serving (one DGX Spark, vLLM)

```bash
vllm serve CocaKova/Agnes-3.0-Flash-Preview-NVFP4 \
  --served-model-name agnes-3.0-flash \
  --max-model-len 131072 --kv-cache-dtype bfloat16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser deepseek_r1
```

Measured on a DGX Spark (GB10, sm_121a): 

| | single stream, thinking off |
|---|---|
| decode, no speculation | 10.6 tok/s (memory-bandwidth floor for ~20 GB of weights) |
| decode, MTP k=3, prose | 15 tok/s (acceptance length ≈ 3.4) |
| decode, MTP k=3, code | 21 tok/s |
| decode, MTP k=3, highly predictable text | 26 tok/s (acceptance ≈ 4.0) |
| cold prefill, 31k-token prompt | 1,650 tok/s |
| KV cache at `--gpu-memory-utilization 0.35` | 206k tokens (bf16 KV) — the rest of the 128 GB stays free |

vLLM 0.20.2-dev tree built for sm_121a, `FlashInferCutlassNvFp4LinearKernel`, Triton/FLA GDN kernels, flashinfer attention.

The chat template is Agnes's own: `chat_template_kwargs: {"reasoning_effort": "low"|"medium"|"xhigh"}` (default **xhigh**; set it lower for agent work) and `enable_thinking: false` to disable reasoning.

## Notes

- This is the Preview open-weight checkpoint, not the production/API Agnes 3.0 Flash listed on Artificial Analysis.
- Delta-rule projections are quantized (same recipe as NVIDIA's Qwen3.8-27B NVFP4 reference); only the conv1d stays bf16.
- Scripts (fold, quantization, verification): https://github.com/CocaKova/agnes-nvfp4
