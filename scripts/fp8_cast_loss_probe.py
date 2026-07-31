"""End-to-end FP8 block-cast loss probe.

Question: is `fp8_block_cast.per_block_cast_to_fp8` lossless enough to serve?
Answer it the honest way — not by isolated per-tensor error, but by the change
in the model's *output logits* when every linear weight is replaced by its
quantize->dequantize round-trip (exactly what the CUDA loader does at serve).

Metrics per prompt:
  - top-1 agreement: does argmax(logits) still pick the same next token?
  - top-5 agreement: is the BF16 top-1 still in the FP8 top-5?
  - logit cosine / relative L2 on the final-position logit vector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp8_block_cast import per_block_cast_to_fp8  # noqa: E402

BLOCK = 128


def dequant(fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of per_block_cast: pad, multiply by block scale, slice. Mirrors
    the loader's dequant (w ~= fp8 * sf)."""
    m, n = fp8.shape
    pm, pn = ((m + BLOCK - 1) // BLOCK) * BLOCK, ((n + BLOCK - 1) // BLOCK) * BLOCK
    padded = torch.zeros(pm, pn, dtype=torch.float32)
    padded[:m, :n] = fp8.float()
    view = padded.view(pm // BLOCK, BLOCK, pn // BLOCK, BLOCK)
    deq = (view * sf.float().view(pm // BLOCK, 1, pn // BLOCK, 1)).reshape(pm, pn)
    return deq[:m, :n]


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.eval()

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Q: What is 17 times 23? A:",
        "The three primary colors are",
        "In 1969, humans first landed on the",
    ]
    # Capture reference (clean BF16) logits BEFORE the in-place quant, so only
    # one model is ever resident — storing 5 tiny logit vectors, not a 2nd model.
    ref_ids = [tok(p, return_tensors="pt").input_ids for p in prompts]
    with torch.no_grad():
        ref_logits = [model(ids).logits[0, -1].float() for ids in ref_ids]

    # Round-trip every 2D linear weight in place: this is the served model.
    # DeepSeek FP8 scope excludes lm_head/embed (they set logits/inputs directly);
    # --scope deepseek skips lm_head to match. Per-tensor error is logged to find
    # which layers dominate the logit shift.
    scope = sys.argv[2] if len(sys.argv) > 2 else "all"
    skip = ("embed",) if scope == "all" else ("embed", "lm_head")
    n_q, worst, errs = 0, 0.0, []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 2 and name.endswith(".weight") and not any(s in name for s in skip):
                w = p.data.cpu().float()
                fp8, sf = per_block_cast_to_fp8(w)
                deq = dequant(fp8, sf)
                rel = ((deq - w).norm() / w.norm()).item()
                worst = max(worst, rel)
                errs.append((rel, name))
                p.data.copy_(deq.to(p.dtype).to(p.device))
                n_q += 1
    errs.sort(reverse=True)
    print(f"scope={scope}: quantized {n_q} linear weights, worst rel-L2 = {worst:.4f}")
    print("  top-3 noisiest tensors:")
    for rel, name in errs[:3]:
        print(f"    {rel:.4f}  {name}")

    top1_hits = top5_hits = 0
    cos_sum = rel_sum = 0.0
    for prompt, ids, lr in zip(prompts, ref_ids, ref_logits):
        with torch.no_grad():
            lq = model(ids).logits[0, -1].float()
        aq, ar = lq.argmax().item(), lr.argmax().item()
        top1_hits += aq == ar
        top5_hits += ar in lq.topk(5).indices.tolist()
        cos_sum += torch.nn.functional.cosine_similarity(lq, lr, dim=0).item()
        rel_sum += ((lq - lr).norm() / lr.norm()).item()
        tag = "OK " if aq == ar else "DIFF"
        print(f"  [{tag}] {prompt!r:40} bf16->{tok.decode([ar])!r} fp8->{tok.decode([aq])!r}")

    n = len(prompts)
    print(f"\ntop-1 agreement {top1_hits}/{n}  top-5 {top5_hits}/{n}  "
          f"logit cos {cos_sum/n:.5f}  rel-L2 {rel_sum/n:.4f}")


if __name__ == "__main__":
    main()
