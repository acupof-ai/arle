#!/usr/bin/env python3
"""HF transformers greedy reference for CUDA parity verification.

Gold ground-truth continuation of RAW token ids (no tokenizer/chat-template), so
it can be compared directly against the rewrite's `infer-cuda` greedy output for
the same ids + model. Transfer this file to the pod and run it (avoids the
shell-heredoc escaping that breaks one-liner `python3 -c` over the tunnel).

Usage:
    python3 hf_greedy_ref.py <model_dir> [max_new_tokens] [id0,id1,...]

Defaults: model=/data01/models/Qwen3-0.6B, max_new=16,
ids=[785,6722,315,9625,374]  ("The capital of France is" for Qwen3).

Prints one line:  HF_GREEDY_REF {"model":..., "prompt_ids":[...], "new_ids":[...]}
Compare `new_ids` against the new infer-cuda greedy tokens (must match exactly).
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM

model = sys.argv[1] if len(sys.argv) > 1 else "/data01/models/Qwen3-0.6B"
max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 16
ids = (
    [int(x) for x in sys.argv[3].split(",")]
    if len(sys.argv) > 3
    else [785, 6722, 315, 9625, 374]
)

m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda().eval()
inp = torch.tensor([ids], device="cuda")
with torch.no_grad():
    out = m.generate(inp, max_new_tokens=max_new, do_sample=False, num_beams=1)
new_ids = out[0].tolist()[len(ids):]

print("HF_GREEDY_REF", json.dumps({"model": model, "prompt_ids": ids, "new_ids": new_ids}))
