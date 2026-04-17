# Qwen3.5 DFlash "acceptance ceiling" was a verify-cost misdiagnosis

## Context

Track 2 of the Qwen3.5 DFlash plan: lift single-session throughput out
of the ~13 tok/s regression vs ~75 tok/s baseline on 4-bit Qwen3.5-4B.

Initial hypothesis: acceptance rate is capped at ~28% on this 4-bit
target, and the only way forward is custom Metal kernels that lower the
per-verify cost. This doc originally concluded:

> The real path forward is the project plan's Phase 1 (custom Metal
> kernels: `gated_delta_with_tape`, `tape_replay`, `batched_sdpa_2pass`).
> That work is sized at 200+ LOC of C++ kernels and is a Codex-led
> project.

**That conclusion was wrong.** The verify-cost ceiling was an artifact
of the implementation, not the kernels.

## Actual root cause

The C++ verify loop ran as 16 sequential `S=1` `cpp_model.step()` calls
— full kernel launch + dispatch + eval per token, 16× per block. The
reference `dflash-mlx` implementation runs verify as a **single
`S=16` block forward**, amortizing the launch/eval overhead and letting
`fast::scaled_dot_product_attention` use causal mode once instead of
bare single-token SDPA 16 times.

We already had all the primitives to do this. The cost wasn't intrinsic
to the kernels — it was sitting inside our step loop.

## Fix

1. **`qwen35_compiled_block_verify`** (new C++ FFI,
   `crates/mlx-sys/src/mlx_qwen35_model.cpp`) — runs the compiled model
   with `current_seq_len = block_size` and `current_last_logits_only =
   false`. Outputs `[logits_[1,B,V], kv_caches..., gdr/conv pairs...]`
   in the same flat layout the Rust side already consumes.
2. **`Qwen35StepDriver::step_block`** (new Rust wrapper,
   `infer/src/backend/metal/qwen35.rs`) — single FFI call per block.
3. **`qwen35_dflash_speculative_block`** (`infer/src/backend/metal/dflash.rs`)
   swapped the 16× `step()` loop for one `step_block(S=16)` call, then
   uses the existing tape/snapshot rollback on partial reject.
4. **kv_flat capacity growth on DFlash path** (`infer/src/backend/metal/request_state.rs`).
   `import_prefix_snapshot` downsizes `kv_capacity` to the replay
   driver's 256-entry allocation; the DFlash decode path bypassed
   `run_step`'s `ensure_capacity`. Once `cache_len + block_size` crossed
   the snapshot's capacity, `slice_update(k_cache, k, [0,0,cache_pos,0],
   [B,nkv,cache_pos+16,hd])` produced a malformed lazy graph and the
   forward died with `Shapes (1,4,16,256) and (1,4,N,256) cannot be
   broadcast`. Fix: grow `kv_capacity` to `cache_len + block_size`
   before entering the speculative block. Mirrors the pattern already
   used in the multi-token prefill path (`request_state.rs:2747`) and
   single-token decode (`request_state.rs:2432`).

## Numbers (post-fix, 4-bit Qwen3.5-4B, M4 Max, Metal)

Three identical 256-token requests plus 4× / 8× concurrent:

| Workload         | tok/s (aggregate) | per-request |
|------------------|-------------------|-------------|
| single (run 1)   | 46.8              | 46.8        |
| single (run 2)   | 46.9              | 46.9        |
| single (run 3)   | 47.2              | 47.2        |
| 4× concurrent    | 153.1             | 38.4        |
| 8× concurrent    | 154.7             | 28.7        |

Previous state (same hardware / quant / prompt):

| Workload      | Pre-block_verify | Post-block_verify (pre-capacity-fix) |
|---------------|------------------|--------------------------------------|
| single        | 13.3             | 42.1 (broke on 2nd repeat request)   |
| 4× concurrent | 148.1            | 144.9                                |
| 8× concurrent | 143.9            | 143.6                                |

Direct A/B against plain decode on the same binary (omit
`--dflash-draft-model`):

| Workload        | Plain decode | DFlash ON | DFlash delta |
|-----------------|--------------|-----------|--------------|
| single (run 1)  | 72.5         | 47.3      | −35%         |
| single (run 2)  | 73.0         | 47.0      | −36%         |
| single (run 3)  | 73.0         | 47.7      | −35%         |
| 4× concurrent   | 159.0        | 154.5     | −3%          |
| 8× concurrent   | 156.1        | 150.1     | −4%          |

Important: the 3.5× is vs the broken DFlash baseline. **DFlash is
still a −35% single-session regression vs plain decode on this quant.**
Concurrent parity is from Track-1 auto-downgrade, not speculative wins.

The speculative math still does not pencil out: at ~28% acceptance a
16-token block nets ~4.5 tokens, so we need `T_S16 < 4.5 × T_S1`.
Measured ratio is closer to `T_S16 ≈ 7 × T_S1` — the S=16 forward is
only ~2.3× faster per-token than 16 S=1 forwards. The GDR
linear-attention recurrence is sequential in the time axis, so the
S=16 savings come entirely from the 8 full-attn layers; the 24 GDR
layers still do per-step work inside the compiled block_verify.

## Rule

**Before concluding an architectural ceiling, measure per-step overhead
against equivalent batched work.** Acceptance rate alone does not tell
you whether speculative decode can be profitable — the verify step has
to amortize more compute than plain decode does per accepted token.
When the verify step is 16× single-token work and the plain decode is
one step, no acceptance rate will rescue the ratio. Look at the shape
of the work, not just the win rate.

**Corollary:** "custom Metal kernels are the only path forward" is a
common misread when the real issue is that our driver is launching
kernels one token at a time. Check the batching of the existing
primitives before scoping a new kernel.

## Follow-ups

- Acceptance still sits around 28% for 4-bit natural-language prompts.
  Not a blocker now that verify cost is amortized, but there's still
  headroom — a better draft model on this quant/prompt distribution
  could push single-session toward baseline parity.
- `dflash-mlx` reference's `batched_sdpa_2pass` remains interesting for
  very large query_len, but is not on the critical path for Qwen3.5.
