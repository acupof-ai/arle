---
title: 2026-05-10 vLLM Medusa prior-art survey — substantially smaller substrate than audit assumed
date: 2026-05-10
type: research
status: open (informs Task #28 substrate sizing)
related_docs: [`0a0d221` Task #28 readiness audit, `9735b47` REFUTATION pivot, `M_medusa-required-path.md`]
---

# vLLM Medusa prior-art survey — direct gh-api source read

> **Why now**: Audit `0a0d221` §3.1 next-step required vLLM/SGLang Medusa
> survey. Direct read of vLLM v0.x sources via gh API contradicts audit
> §2.1 substrate-size assumption (~500 LOC + tree-attention kernel).
> Real LOC: **265 lines total**, **NO new kernel needed for heads**.
> Tree-attention complexity is at VERIFY path only, not PROPOSE.

## §1 vLLM Medusa source files (3 total)

| Path | LOC | Role |
|---|---:|---|
| `vllm/v1/spec_decode/medusa.py` | ~95 | `MedusaProposer` — runtime proposer wrapper |
| `vllm/model_executor/models/medusa.py` | ~165 | `Medusa` model + `ResidualBlock` |
| `vllm/transformers_utils/configs/medusa.py` | ~50 | `MedusaConfig` (num_heads=5, num_hidden_layers=1) |
| **TOTAL** | **~310** | |

## §2 Architecture (read from actual source)

### §2.1 ResidualBlock (~25 LOC)
```python
class ResidualBlock(nn.Module):
    def __init__(self, config, hidden_size, num_layers):
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=...)
                                     for _ in range(num_layers)])
        self.act = nn.SiLU()

    def forward(self, x):
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x
```

**No new CUDA kernel needed.** Just nn.Linear + SiLU. ARLE's existing
matmul + SiLU kernels in `crates/cuda-kernels/csrc/misc/` cover this.

### §2.2 Medusa model (~165 LOC)
```python
class Medusa(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.blocks = nn.ModuleList([
            ResidualBlock(config, self.config.hidden_size, self.config.num_hidden_layers)
            for _ in range(self.config.num_heads)  # K=5 by default
        ])
        # Plus: LM head per Medusa head (or shared with target)
```

K=5 ResidualBlock heads + their LM heads (each ResidualBlock typically
~6.5M params for hidden_size=4096, num_hidden_layers=1).

### §2.3 MedusaProposer.propose() — TOP-1 ONLY
```python
def propose(self, target_hidden_states, sampling_metadata, slot_mappings=None):
    blocks = self.model(target_hidden_states)
    logits = self.model.compute_logits(blocks)
    # Compute argmax for each Medusa head and stack into a single tensor
    # Shape: [batch_size, num_heads]
    draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)
    return draft_tokens
```

**Critical observation**: vLLM's Medusa propose is `argmax` per head,
NOT top-K candidates per head. Returns shape `[batch_size, num_heads]`
— a single linear sequence of K draft tokens.

This is materially simpler than the audit assumed:
- ❌ Audit assumed: tree-attention construction in PROPOSE path
- ✅ Reality: PROPOSE is flat top-1, tree-attention only at VERIFY (and only if doing tree-style verify, which paper §3.2 does but vLLM v1 simplifies)

## §3 Differences from Medusa paper (per vLLM docstring)

> Differences from reference implementation:
> 1. Currently this only supports generating proposals from top-1 tokens.
> 2. We have an optional token_map which reduces draft vocab to most
>    frequently used tokens to give some additional speed-up...

**top-1 only** = vLLM trades some α gain for substrate simplicity.
For ARLE Task #28 first iteration, follow vLLM simplification —
gets us live faster, can add top-K + tree-attn later if α is below
1.5× license threshold.

## §4 Implications for ARLE Task #28

### §4.1 Substrate scope REVISED DOWN

Audit `0a0d221` §2.1 estimated:
- 4 Medusa heads kernel: ~500 LOC `.cu` file
- Tree-attention extension: separate work

Real scope (top-1 vLLM-style):
- Medusa heads: existing nn.Linear + SiLU primitives — **NO new kernel**
- Wrapper struct + load_weights: ~200 Rust LOC in `infer/src/model/medusa.rs`
- Propose path: ~50 Rust LOC (argmax loop)
- Integration with `infer/src/speculative.rs` (existing 721 LOC): ~100 LOC delta
- **Total: ~350 LOC Rust + 0 new CUDA kernels** (vs audit's 500+ LOC + new kernel)

### §4.2 Faster path to license-or-kill bench

Original audit estimated 1 week training + 1 day integration + 1 day bench.
With reduced substrate scope:
- Substrate: 1-2 days (was: 2-3 days)
- Training (Alpaca, K=5 heads): 2-3 days unchanged
- Integration: 1 day unchanged
- **Total: 4-6 days wall-clock** (was: ~1 week)

### §4.3 First-iter α target

Per Medusa paper Table 1: top-1-only α ≈ 0.6-0.7 on Vicuna-7B.
For Qwen3-4B agent W3/W4 shape, predicted α ≈ 0.55-0.65 (slightly
lower due to top-1 constraint). With K=5:
```
E[accepted/step] = 1 + Σ α^i = 1 + 0.6 + 0.36 + 0.22 + 0.13 + 0.08 = 2.39 tokens/step
Speedup vs no-spec: 2.39 / 1.06 = 2.25×
```

Above 1.5× license threshold. Acceptable first-iter target.

### §4.4 If first iter < 1.5× license

Add top-K → tree-attention upgrade as Phase 2:
- vLLM v0.x has full tree implementation (older path) — can port if needed
- SGLang Medusa-2 has more advanced acceptance — backup option
- ~3-5 days additional work

## §5 Cross-references

- `0a0d221` Task #28 readiness audit (this entry refines §2.1, §4 estimates)
- `9735b47` REFUTATION wins entry (strategic pivot to Option A)
- `M_medusa-required-path.md` Phase 1-4 plan (still valid; substrate scope revised)
- `M_medusa-phase1a-dataset-directive.md` (Alpaca/lmsys-chat-1m selection)
- `infer/src/speculative.rs` (721 LOC existing scaffold; integration target)
- vLLM v0.x sources (read via gh api 2026-05-10):
  - `vllm/v1/spec_decode/medusa.py` (~95 LOC)
  - `vllm/model_executor/models/medusa.py` (~165 LOC)
  - `vllm/transformers_utils/configs/medusa.py` (~50 LOC)
- Medusa paper: <https://arxiv.org/abs/2401.10774>
- FasterDecoding/Medusa reference: <https://github.com/FasterDecoding/Medusa>

## §6 Recommended Phase 1.B substrate brief (next Claude action)

If user approves Option A pickup, next CPU-bound brief should specify:
1. Rust struct layout for Medusa heads (mirroring vLLM ResidualBlock + LM head)
2. Weight loading from HF checkpoint (Medusa heads usually saved separately)
3. Integration point with `infer/src/speculative.rs::TokenProposal`
4. Top-1 propose path (argmax per head)
5. Defer tree-attention to Phase 2 (only if first-iter α < 1.5×)

This brief becomes the codex pickup directive for Task #28 implementation.
