# INT8 KV → KIVI per-channel K — bit-identical with BF16

## Context

The 2026-05-27 V100 Qwen3.5-4B audit (see
[`2026-05-27-v100-kv-precision-parity-qwen35-4b.md`](2026-05-27-v100-kv-precision-parity-qwen35-4b.md))
locked in the first non-degenerate KV-quant baseline on this
codebase and surfaced a hard signal: INT8 KV diverged from BF16
deterministically at step 1 across both prompts, mean_match=0.0938.
FP8 in the same audit was bit-identical (mean_match=1.0000).

Initial framing was "INT8 is real quantization noise, not a kernel
bug". The user pushed back: same prompt, same model, same greedy
decode — a 90% mismatch is not noise.

## What worked

Located the discriminator in `crates/cuda-kernels/src/paged_kv.rs:495`:

```rust
let (k_static_scales, k_kivi_calibrated) =
    if matches!(format, KVFormat::FP8E4M3) && pool_bytes_per_layer > 0 {
        // alloc per-(kv_head, head_dim) static scales — only for FP8
    } else {
        (None, Vec::new())   // ← INT8 lands here
    };
```

FP8 had been wired to KIVI's asymmetric scheme (per-channel K +
per-token V) at commit `8c6d92db` (2026-05-26). INT8 still ran on
per-(token, head) absmax for K, which flattens the outlier channels
that K activations in modern dense transformers exhibit (the KIVI
ICML 2024 paper's central finding).

Direct mirror port from FP8 to INT8:

1. **paged_kv.rs gate**: `FP8E4M3` → `FP8E4M3 | INT8`. Pool now
   allocates a `[num_kv_heads, head_dim]` f32 scale table per layer
   for both formats. Memory cost: ~576 KB for Qwen3-4B-class, negligible
   vs the 19.9 GB pool.
2. **CSRC kernels** (in `csrc/kv/kv_quant.cu` and
   `csrc/attention/decode_attention_quantized.cu`):
   - `quantize_paged_kv_int8_per_channel_kernel` — port of the FP8
     template with `int8_t` dst, `__float2int_rn` + clamp to ±127.
   - `finalize_k_per_channel_scales_int8_kernel` — same as FP8 but
     divides accumulated absmax by 127 instead of 448.
   - `decode_attention_int8_per_channel_k_partial_kernel` — same
     cp.async-pipelined smem tiling as the per-token INT8 sibling,
     but K scales are pre-loaded into `k_scale_reg[EPT]` registers
     once at kernel entry from the static `[num_kv_heads, head_dim]`
     table; V scales stay per-(row, head) via the existing smem
     async load.
3. **FFI + Rust wrappers** mirror the FP8 KIVI signatures exactly.
4. **Dispatch wiring** in `qwen3/prefill.rs` + `qwen3/batch_decode.rs`
   + `qwen35/prefill.rs` + `qwen35/batch_decode.rs`: route INT8 K
   through the per-channel path when `k_static_scales_ptr` is Some.
5. **V100 audit re-run** post-fix:

   | precision | pre-fix mean_match | post-fix mean_match |
   |---|---:|---:|
   | bf16 (ref) | 1.0000 | 1.0000 |
   | int8 | **0.0938** | **1.0000** |
   | fp8 | 1.0000 | 1.0000 |
   | tq4 | n/a (sm_70 unsup) | n/a |

   INT8 decode tokens are now token-by-token bit-identical with BF16
   across 32 generated tokens / 2 prompts. Gate passes
   (`gate=0.99, passed=true`).

After validation, the per-(token, head) K kernels were retired in
the same effort (commit `ba74dd49`): 1052 lines of dead CSRC + Rust
+ FFI + smoke test deleted. INT8 and FP8 now share a single
production path (per-channel K), with V always per-(row, head) per
KIVI's asymmetric scheme. The retire pass caught one straggler
(qwen35/prefill.rs FP8 arm was still calling the per-token K
quantize after the rest of the dispatch had moved to per-channel K
— second attempt with FP8 calibration in place restored
`fp8 mean_match=1.0000`).

## Rule

**"Real quantization noise" is the lazy diagnosis.** When two
precisions show structurally different mismatch under matched
hardware + model + prompt + greedy decode, the discriminator is
almost always a missing optimization in the path — not a unit-test-
clean kernel producing fundamentally noisier output. The first
question after a mean_match split should be "what does the passing
path do differently?" before any conclusion about quant fundamentals.

For LLM KV cache specifically: **K's outlier-channel structure
requires per-channel calibration to recover BF16 quality at INT8/FP8
precision**. This is the KIVI paper's central finding; per-(token,
head) absmax is a defensible default for V but is documented to
underperform on K in modern dense transformers.

After validating an alternative path, **retire the dead one** —
don't leave both behind as a `match` arm. Codebase carry tax adds
up: 1052 lines of "we keep this in case" is fuel for the next
regression.

## Related

- [`docs/plans/2026-05-27-int8-kv-kivi-per-channel.md`](../../plans/2026-05-27-int8-kv-kivi-per-channel.md)
  — the planning doc that became this fix; mark as DONE.
- [`docs/experience/wins/2026-05-27-v100-kv-precision-parity-qwen35-4b.md`](2026-05-27-v100-kv-precision-parity-qwen35-4b.md)
  — the V100 audit that surfaced the gap.
- [`docs/experience/errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`](../errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md)
  — the prior FP8 "catastrophic" claim that was a degenerate-baseline
  test artifact, NOT the same bug as INT8's real per-token K drift
  here.
- Commit `8c6d92db` — original FP8 KIVI landing (V1, gated to FP8).
- Commit `8afecffe` — INT8 KIVI feat (gate extension + new kernels
  + dispatch wiring).
- Commit `ba74dd49` — retire per-token K kernels; single path.
- KIVI paper: <https://arxiv.org/abs/2402.02750>.
