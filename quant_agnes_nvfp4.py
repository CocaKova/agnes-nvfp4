#!/usr/bin/env python3
"""NVFP4 (W4A4, group 16) quantization of Agnes-3.0-Flash Preview (folded to the Qwen3.5 graph).

Recipe is a byte-for-byte match of the reference build for this same
architecture (models/qwen3.8-27b-nvfp4-mtp/recipe.yaml): vision tower,
lm_head, DeltaNet conv1d and the whole MTP head stay bf16, everything
else goes NVFP4.

The model MUST be constructed explicitly as Qwen3_5ForConditionalGeneration.
Letting oneshot() resolve the class itself yields Qwen3_5ForCausalLM -- a
text-only graph with 0 vision and 0 MTP tensors, so the visual/mtp ignore
patterns below match nothing and both heads are silently dropped from the
output. Verify the result by counting tensors, not by trusting the log.

The MTP ignore entries are load-bearing: if the server treats the bf16
draft head as NVFP4 the draft breaks silently — 0% acceptance and
slower than running with no MTP at all.
"""
import gc, json, os, sys, torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL   = "/home/cocakova/models/agnes-3.0-flash-qwen35-bf16"
OUTPUT  = "/home/cocakova/models/agnes-3.0-flash-nvfp4"
MAX_SEQ = int(os.environ.get("MAX_SEQ", 4096))  # ref build used 8192 across 8 GPUs; halved for the single-GB10 unified-memory box
N_CAL   = int(os.environ.get("N_CAL", 32))

print(f"[1/4] Tokenizer from {MODEL}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.model_max_length = MAX_SEQ

print(f"[2/4] Calibration ({N_CAL} x {MAX_SEQ} tok, streamed)", flush=True)
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)

# llm-compressor 0.12 inspects `dataset.column_names`, so a bare generator is
# rejected (the older quantize-qwen3-32b script predates that check). Materialize
# a real Dataset instead -- 32 x 4096 tok is a few MB, nothing to stream.
def build_calibration(ds, n, max_len):
    rows, count = [], 0
    for row in ds:
        if count >= n:
            break
        msgs = row.get("messages", [])
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                chat_template_kwargs={"enable_thinking": False},
            )
        except Exception:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=max_len)
        if len(enc["input_ids"]) < 16:
            continue
        rows.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        count += 1
        if count % 8 == 0:
            print(f"    calibration sample {count}/{n}", flush=True)
    return Dataset.from_list(rows)

calib = build_calibration(ds, N_CAL, MAX_SEQ)
print(f"    calibration dataset ready: {calib.num_rows} rows, columns {calib.column_names}", flush=True)

gc.collect()
torch.cuda.empty_cache()

print("[3/4] Loading full multimodal graph (vision + MTP)", flush=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cpu",
)
nvis = sum(1 for n, _ in model.named_parameters() if "visual" in n)
nmtp = sum(1 for n, _ in model.named_parameters() if n.startswith("mtp"))
print(f"    loaded {type(model).__name__}: {nvis} vision params, {nmtp} mtp params", flush=True)
# nmtp == 0 is EXPECTED and not an error: transformers' ConditionalGeneration
# class has no MTP submodule, so the draft head can never be present here. It
# is lifted from the bf16 source afterwards by wire-mtp-into-nvfp4.py. Only a
# missing vision tower means we are about to build a crippled model.
if nvis == 0:
    sys.exit("ABORT: vision tower missing from the loaded graph -- refusing to produce a text-only build")

print("[3/4] oneshot NVFP4 ...", flush=True)
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head", "re:.*visual.*", "re:.*conv1d.*", "re:.*mtp.*"],
)
# processor=tokenizer is deliberate. AutoProcessor for this arch pulls in
# Qwen3VLVideoProcessor, which needs torchvision -- absent from quant-env.
# Calibration is text-only and the vision tower is in `ignore`, so the
# tokenizer is the whole processor we need; installing torchvision here
# would risk dragging in a second torch build.
oneshot(
    model=model,
    processor=tokenizer,
    recipe=recipe,
    output_dir=OUTPUT,
    dataset=calib,
    max_seq_length=MAX_SEQ,
    num_calibration_samples=N_CAL,
)

gc.collect()
torch.cuda.empty_cache()
print(f"[4/4] Done -> {OUTPUT}", flush=True)
