---
title: Path B-Phase2' P0.B PPL gate — existing eval infra inventory + recommended codex path
date: 2026-05-10
type: research
status: inventory-for-codex-pickup-post-p0a
---

# Path B-Phase2' P0.B PPL gate — existing eval infra inventory + recommended codex path

> Codex is mid-P0.A cutlass FP8 GEMM spike (Working 2m+ on
> ada_fp8_gemm_example.cu source reading). Surveying existing PPL
> eval infra so P0.B (BF16→FP8 quant accuracy gate) can pick up
> cleanly post-P0.A. Tick deliverable: durable inventory + execution
> path, no duplicated work.

## §0 Existing scripts (verified raw `ls`/`grep` this tick)

```
scripts/eval_ppl.py              231 LOC  — KV-format PPL via server logprobs
scripts/eval_topk_regression.py           — top-K accuracy (different gate)
scripts/gen_arle_longctx_eval.py          — long-context eval data gen
crates/train/src/eval_lm.rs               — train-side eval harness (Rust)
```

`scripts/eval_ppl.py` is the natural P0.B base. Already does:

```python
# Function inventory (raw grep this tick)
11: import argparse
25: def load_dataset_texts(name, max_samples=20):     # wikitext + humaneval
73: def start_server(kv_dtype=None):                  # supports --kv-cache-dtype
100: def stop_server(proc):
109: def collect_logprobs(prompt, max_tokens=50):
140: def compute_ppl(logprobs):
147: def eval_format(label, dtype, dataset_texts, max_tokens):
164: def main():
```

Server start command in eval_ppl.py:73 uses
`target/release/infer --model-path <X> --port 8090 --num-slots 1`
plus optional `--kv-cache-dtype <bf16|fp8|int8>`.

**Already designed for sequential A/B between quant formats.**

## §1 P0.B execution paths (two viable approaches)

### Path α — Server-side (depends on P0.A passing + Phase 2'.1 substrate)

If P0.A licenses cutlass FP8 GEMM, then Phase 2'.1 (BF16→FP8 act quant
kernel ~60 LOC) lands as runtime substrate. Then `eval_ppl.py` can be
extended to dispatch via a new flag like `--act-quant <bf16|fp8|int8>`
and reuse the existing format-A/B pattern.

**Pros**: end-to-end runtime measurement (matches production)
**Cons**: requires P0.A + Phase 2'.1 to land first → not parallel-able with P0.A

### Path β — Offline standalone (independent of P0.A)

Write a new `scripts/ppl_simulate_act_quant.py` (~150-200 LOC) that:

1. Loads Qwen3-4B BF16 weights via `safetensors` Python
2. Replays forward pass on Wikitext-103 prompts using
   `transformers.AutoModelForCausalLM` (PyTorch path — pure CPU/GPU
   reference, NO ARLE runtime dep)
3. At each linear layer, simulates activation quantization:
   - Baseline: INT8 per-channel scale (matches current ARLE W4A8)
   - Treatment: FP8 e4m3 per-channel scale
4. Computes PPL Δ between baseline and treatment

**Pros**: parallel-able with P0.A (no runtime dep), accuracy-only signal
**Cons**: doesn't measure end-to-end runtime PPL; PyTorch numerics may
diverge from ARLE inference numerics by small margins (test-only ±2%
PPL noise is typical and acceptable for a license gate)

**Recommendation**: codex picks Path β if P0.A is taking longer than
~4h, OR Path α if P0.A clears early. Both are valid for the gate
threshold (PPL Δ ≤ 0.5).

## §2 Suggested implementation outline for Path β (if codex picks it up)

```python
# scripts/ppl_simulate_act_quant.py (new, ~150-200 LOC)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = "infer/models/Qwen3-4B"

def quantize_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Simulate per-channel FP8 e4m3 quant + dequant."""
    # e4m3 dynamic range ~ ±448
    scale = x.abs().amax(dim=-1, keepdim=True) / 448.0
    q = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.to(torch.bfloat16) * scale

def quantize_to_int8(x: torch.Tensor) -> torch.Tensor:
    """Simulate per-channel INT8 quant + dequant (current ARLE W4A8)."""
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
    q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return q.to(torch.bfloat16) * scale

def hook_act_quant(module, input, output, quant_fn):
    """Forward hook to quant activations entering/leaving Linear layers."""
    if isinstance(input, tuple):
        input = (quant_fn(input[0]),) + input[1:]
    return input

def eval_ppl(model, tokenizer, texts, max_tokens=200):
    """Compute mean per-token NLL → PPL on a list of texts."""
    nlls = []
    for text in texts:
        ids = tokenizer.encode(text, return_tensors="pt").cuda()[:, :max_tokens]
        with torch.no_grad():
            outputs = model(ids, labels=ids)
        nlls.append(outputs.loss.item())
    return torch.tensor(nlls).exp().mean().item()

def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    texts = [r["text"] for r in load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test")
        if len(r["text"].strip()) > 100][:50]

    # Baseline (no quant)
    base_ppl = eval_ppl(model, tokenizer, texts)

    # INT8 quant
    int8_hooks = [m.register_forward_pre_hook(
        lambda m, i: (quantize_to_int8(i[0]),))
        for m in model.modules() if isinstance(m, torch.nn.Linear)]
    int8_ppl = eval_ppl(model, tokenizer, texts)
    for h in int8_hooks: h.remove()

    # FP8 e4m3 quant
    fp8_hooks = [m.register_forward_pre_hook(
        lambda m, i: (quantize_to_fp8_e4m3(i[0]),))
        for m in model.modules() if isinstance(m, torch.nn.Linear)]
    fp8_ppl = eval_ppl(model, tokenizer, texts)
    for h in fp8_hooks: h.remove()

    print(f"Baseline (BF16): {base_ppl:.3f}")
    print(f"INT8 simulated:  {int8_ppl:.3f}  (Δ vs base = {int8_ppl - base_ppl:+.3f})")
    print(f"FP8 e4m3:        {fp8_ppl:.3f}  (Δ vs base = {fp8_ppl - base_ppl:+.3f})")
    print(f"FP8 vs INT8:     Δ = {fp8_ppl - int8_ppl:+.3f}  (gate ≤ 0.5)")
```

Wall time for codex if picked up: ~2h (write + run + interpret). Can
run in parallel with P0.A cutlass smoke (different processes, no GPU
contention since PyTorch BF16 forward at batch=1 uses ~8GB while ARLE
release inference uses ~15GB — both fit on 16GB if scheduled
sequentially or with batch=1).

## §3 Decision matrix for P0.B post-P0.A

| P0.A outcome | Recommended P0.B |
|---|---|
| Pass ≥3× speedup (license) | Path α — extend eval_ppl.py with `--act-quant` flag (after Phase 2'.1 lands) |
| Mid-zone 2-3× speedup (uncertain) | Path β first (parallel) — if FP8 PPL Δ > 0.5, KILL Phase 2' regardless of perf |
| Kill ≤2× speedup | Skip P0.B; pivot to Phase 2 multi-shape spec OR Phase 1 dequant.h port |

## §4 Cross-references

- Phase 0 brief: `docs/research/2026-05-10-path-b-phase2-prime-phase0-brief-codex-kickoff.md` (5a7a28b)
- Cutlass sm_89 unstick: `docs/research/2026-05-10-cutlass-sm89-fp8-template-found-codex-unstick.md` (d5a6679)
- M_quant magnitude plan: `docs/plans/M_quant-fp8-w4-magnitude-path.md`
- Existing PPL eval script: `scripts/eval_ppl.py` (231 LOC)
- Train-side eval: `crates/train/src/eval_lm.rs`
- Top-K eval: `scripts/eval_topk_regression.py`

## §5 Status

P0.B substrate inventoried. Two viable paths (α server-side requires
P0.A pass + Phase 2'.1 land; β offline standalone parallel-able).
Codex can pick up either post-P0.A based on cutlass smoke outcome.
Wall time ~2h either path.

This tick = inventory only, no script written by Claude (avoids
duplication — codex may take Path α via existing eval_ppl.py, in
which case Path β draft would be wasted work).
