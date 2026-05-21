import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B"
OUT = Path("bench-output/2026-05-21-qwen35-9b-tq4-generation-quality")
PROMPTS = [
    "Hello, world! Tell me a short story about a small robot.",
    "Explain on-policy distillation in two sentences.",
    "Write a Python function that returns the Fibonacci sequence up to n.",
]

print(f"loading tokenizer from {MODEL}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
print("loading model with device_map=auto", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "14GiB", "cpu": "80GiB"},
    offload_folder=str(OUT / "hf_offload"),
    low_cpu_mem_usage=True,
)
model.eval()
print(f"model loaded in {time.time() - t0:.3f}s", flush=True)
print(f"hf_device_map={getattr(model, 'hf_device_map', None)}", flush=True)

for idx, prompt in enumerate(PROMPTS, 1):
    inputs = tok(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    # Send input tensors to the first parameter device; accelerate dispatches modules from there.
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    t1 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=None,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.time() - t1
    completion_ids = out[0, input_len:]
    completion = tok.decode(completion_ids, skip_special_tokens=True)
    record = {
        "prompt": prompt,
        "completion": completion,
        "prompt_tokens": int(input_len),
        "completion_tokens": int(completion_ids.numel()),
        "elapsed_seconds": elapsed,
    }
    (OUT / f"pytorch_bf16_completion_{idx}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"--- PyTorch BF16 completion {idx} elapsed={elapsed:.3f}s tokens={completion_ids.numel()} ---", flush=True)
    print(completion, flush=True)
