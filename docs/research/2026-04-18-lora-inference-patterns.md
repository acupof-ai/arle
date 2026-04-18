# LoRA inference patterns across Rust/C++ LLM engines

**Date:** 2026-04-18
**Motivation:** M2b blocker analysis enumerates three options
(CUDA autograd ramp · forward-only runtime GEMM hook · defer). Surveyed
mainstream engines to calibrate which pattern is load-bearing in practice
and whether we missed a cheaper alternative.

## Summary of findings

| Engine | LoRA strategy | Base-weight quant compatible? | Overhead at inference |
|--------|---------------|-------------------------------|----------------------|
| llama.cpp | **Merge-time** (default) or static-runtime via `convert_lora_to_gguf.py` + `--lora` flag | Yes — merge before GGUF quantization | **Zero** after merge; tiny when runtime-applied |
| mistral.rs | **Runtime low-rank GEMM**, adapter layers stay fp16/bf16 even when base is quantized; dynamic adapter swap at runtime | Yes — adapter never quantized | Per-layer extra matmul, hybrid-precision |
| vLLM (Punica/SGMV) | **Runtime batched** — a custom SGMV kernel fuses many tenants' LoRA deltas into one GEMM | Yes (depends on the quant flavor) | ~+2 ms/token for thousands of concurrent adapters, 12× throughput vs naive |

Sources below.

## What this tells us about M2b

Option (a) in `m2b-blocker-analysis.md` is a **production RL** pattern
(train adapters + serve immediately, gradient loop closed). Cost: 10–14 days
CUDA autograd + frozen-view work. Only pays off if that loop is the user story.

Option (b) ("runtime low-rank GEMM hook in `linear.rs`") is the **mistral.rs**
pattern — valuable for multi-adapter swap, adds per-token latency on every
linear it touches. Cost: ~2 days, medium risk because every Qwen3 forward
touches `linear.rs`.

**Option (b′) — not listed in the blocker doc, surfaced by this research:**
**merge-time utility**. Load base weights (safetensors), load LoRA A/B
(safetensors), compute `W' = W + alpha/rank · B @ A`, write a new
safetensors checkpoint. **Zero changes to `linear.rs`.** This is what
llama.cpp does by default, and matches the finding that "merged adapters
compile back into the original weights, adding zero extra inference
overhead." Cost: ~1 day, zero risk to the hot path.

- Pros: doesn't touch `infer/` at all. Qwen3 inference binary unchanged.
  Reversible (just delete the utility). Works even if the user never
  unblocks full CUDA autograd.
- Cons: doesn't support dynamic adapter swapping. Incompatible with
  per-request LoRA routing (each adapter needs its own merged checkpoint
  on disk). Not suitable for multi-tenant serving.
- Who cares: for a single-researcher self-evolve loop (the project's
  stated goal), dynamic swapping is not on the path. You train one
  adapter, merge, serve. That's the mistral.rs + llama.cpp hobbyist
  pattern.

## Revised recommendation

The blocker doc's Option (a)–(c) framing overstated the necessity of a
runtime LoRA hook. For **the project's stated self-evolve user story**,
**option (b′) = merge-time utility** is almost certainly the right first
move:

1. It unblocks the Qwen3-LoRA end-to-end story without touching `linear.rs`.
2. It's compatible with every quant flavor we already ship (Marlin, W4/W8,
   Q4K/Q6K) — as long as the merge happens in fp32 before requantization.
3. It leaves option (a) open for later — the day we genuinely need the
   train↔infer gradient loop, we implement it then.

Option (a) remains the correct choice **only if** the user story is
"real-time policy updates during RL, serve immediately" — which the
project currently does not need (TinyLM RL already works end-to-end via
the checkpoint save/load path).

## Sources

- [mistral.rs ADAPTER_MODELS.md](https://github.com/EricLBuehler/mistral.rs/blob/master/docs/ADAPTER_MODELS.md) — "X-LoRA or LoRA adapter layers will not be quantized, only the base model"; dynamic runtime swap.
- [mistral.rs LoraModelBuilder example](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/lora/main.rs) — runtime adapter mount via builder.
- [llama.cpp adapters wiki](https://deepwiki.com/ggml-org/llama.cpp/3.10-adapters-and-fine-tuning) — merge via `convert_lora_to_gguf.py`; runtime apply for dynamic.
- [llama.cpp merge issue #7062](https://github.com/ggml-org/llama.cpp/issues/7062) — community preferred path is pre-merge before GGUF.
- [llama.cpp LoRA swap discussion #8849](https://github.com/ggml-org/llama.cpp/discussions/8849) — runtime swap exists but merge is default.
- [Punica paper (SGMV)](https://arxiv.org/pdf/2310.18547) — segmented-gather-matrix-vector kernel for multi-tenant.
- [LMSYS S-LoRA blog](https://www.lmsys.org/blog/2023-11-15-slora/) — thousands of adapters at low overhead, multi-tenant scenario.
- [vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/) — Punica integration.
