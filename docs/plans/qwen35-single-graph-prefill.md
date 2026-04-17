# Qwen3.5 full-forward CUDA Graph prefill — Plan

## Status

Active plan as of 2026-04-17. Targets the ~46% peak-throughput gap and
~60% TTFT-p99 gap between infer and sglang on Qwen3.5-4B at L4,
documented in `docs/experience/wins/2026-04-17-sglang-p99-parity-qwen35-4b.md`.

## One-line thesis

sglang captures the entire Qwen3.5 forward (all 32 hybrid layers + norms +
LM head) as a single CUDA Graph during prefill. We launch each kernel
separately — 24 linear-attn layers × ~7 sub-kernels each + 8 full-attn
layers × ~5 sub-kernels each = **~200 individual kernel launches per
forward**. On L4 each launch costs ~2-3µs → 400-600µs of pure overhead
per forward. At steady-state that compounds into the observed gap.

## What changes in the code

### Shape parameterisation

A single graph is captured per `(batch_size, seq_len)` tuple. Realistic
prefill shapes on our benches:
- Qwen3.5-4B sweep: seq_len ∈ {256, 512, 1024, 2048, 4096}; batch=1.
- Decoding already captured at (B, 1) for B=1..10.

For prefill, the space is bigger. Two policies to pick from:
1. **Bucketed capture** — round seq_len up to nearest power of 2 (256,
   512, 1024, 2048, 4096). 5 graphs × 1 batch = 5 captures. Pads query
   with masked tokens past the true length.
2. **On-demand capture + LRU** — capture first time a shape is seen,
   cache up to N=10 graphs, evict LRU. No padding overhead; captures
   amortise across bench runs.

Start with **(1) bucketed**. Simpler, predictable latency, mask cost
is negligible at prefill.

### State buffer handoffs must be contiguous

Qwen3.5's linear-attention layers write recurrent state to a per-row
buffer. Currently, each layer allocates its own intermediate `HiddenStates`
via `DeviceContext::alloc`. For graph capture to work, **all intermediate
tensors must be preallocated once with stable device pointers**. Today
`PrefillBuffers` in `infer/src/model/qwen35/prefill_buffers.rs` already
does this for the main hidden-state path. Missing:

- per-layer recurrent-state scratch (currently allocated inside
  `gated_delta_rule_prefill_chunk_*` callers)
- KV write staging (currently allocated per-layer in `paged_kv_write_*`)
- attention output scratch (FlashInfer tmp_buffer — currently per-call)

Audit: all `DeviceContext::alloc*` calls in the Qwen3.5 prefill path must
become either `PrefillBuffers` fields or early-allocated-once buffers.

### FlashInfer tmp_buffer lifetime

`flashinfer_single_prefill_hd256` takes a `tmp_buf` argument. Currently
a fresh scratch is passed per call. For graph capture, this scratch must
be a fixed device pointer — allocate once in `PrefillBuffers` sized for
the largest captured seq_len.

### Graph capture guard rails

1. **No CPU-GPU sync during capture.** `forward_prefill` currently does
   `ctx.sync()` after logits readback for the first-token sample. During
   capture this must be deferred until after graph launch. Split into
   `forward_prefill_launch` + `readback_first_token` similar to the
   decode path.
2. **No host allocations.** `Vec::with_capacity` inside the forward is
   fine; the tensor data must already be on device with stable ptrs.
3. **No variable control flow.** Every branch inside the captured
   forward must be data-independent (same kernels in same order for all
   requests at the captured shape). Audit the `LayerKind::Linear` vs
   `LayerKind::Full` dispatch — already data-independent per-layer index.

## Files to touch

| File | What changes |
|------|--------------|
| `infer/src/model/qwen35/prefill_buffers.rs` | Add fields for all scratch buffers currently allocated per-call. Size for max seq_len bucket. |
| `infer/src/model/qwen35/prefill.rs` | Drop all per-call `DeviceContext::alloc*`. Use preallocated fields. |
| `infer/src/model/qwen35/forward.rs` | Split `forward_prefill` into `forward_prefill_launch` (pure launch sequence, captureable) + `forward_prefill_readback` (the sync + first-token sample). |
| `infer/src/model/qwen35/batch_decode.rs` | Extend `PrefillGraphCache` paralleling existing decode-graph cache; key by `(seq_len_bucket,)`. |
| `infer/src/scheduler/cuda/prefill.rs` | Call the graph-replay variant when a match exists. Fall back to non-captured path on first call. |
| `crates/infer-cuda-kernels/csrc/attention/flashinfer_prefill_hd256.cu` | Verify `cudaError_t` return does not trigger host sync during capture; may need to inspect FlashInfer wrapper for hidden syncs. |

## Acceptance criteria

1. `cargo build --release -p infer` clean. `cargo check -p infer
   --no-default-features --features cuda,no-cuda` clean.
2. `cargo test --release --test e2e_qwen35` passes — same JSON baselines.
3. Qwen3.5-4B guidellm sweep at `prompt_tokens=4096,output_tokens=256`:
   - **Peak throughput ≥ 115 tok/s** (baseline 91.4 tok/s; sglang 134 tok/s).
     Target closes ≥55% of the 42.6 tok/s gap.
   - **TTFT p99 at 0.135 r/s ≤ 1150ms** (baseline 1359ms; sglang 964ms).
     Target closes ≥50% of the 395ms gap.
4. Qwen3.5 ITL p99 does NOT regress past +0% (baseline advantage 3-8%
   must be preserved).
5. First-prefill latency on a new (batch, seq_len) tuple may be worse
   (capture cost). Amortises after 1st call. Document this in a new
   wins entry.

## Risks and pre-flight checks

- **FlashInfer internal sync**: if `SinglePrefillWithKVCacheDispatched`
  issues a device sync (cudaStreamSynchronize or equivalent), capture
  will fail with `cudaErrorStreamCaptureInvalidated`. Pre-check by
  trying a tiny capture of just `flashinfer_single_prefill_hd256` on a
  dev box before wiring the full forward.
- **Recurrent state clobber**: if the same `Qwen35State` scratch is
  used across replays before the previous replay drains, graphs will
  overlap reads/writes. Ensure one-inflight-prefill-per-slot invariant.
- **seq_len masking**: Bucketed capture pads to the next bucket. The
  mask must suppress attention to padding tokens. FlashInfer's
  `use_custom_mask=false` + `kCausal` handles causal masking but NOT
  arbitrary padding. Two options:
  1. Enable `use_custom_mask=true` with a mask that zeros padded rows.
     Adds per-call mask build cost.
  2. Keep qo_len = true seq_len, preallocate buffer for max bucket,
     only run attn over true length. FlashInfer accepts qo_len < buffer
     capacity, so this is the cleaner path.

  Go with (2).

## Benchmark methodology for this plan

1. **Before-snapshot** — already have, 2026-04-17 Qwen3.5-4B parity.
2. **After-snapshot** — `scripts/bench_guidellm.sh qwen35-4b-infer-l4-fullgraph`
   followed by a paired parity run vs sglang with a delta table showing
   peak tok/s, TTFT p99 at each matched rate, ITL p99 at each matched
   rate.
3. **Graph-capture overhead bench** — one-off measurement: time
   `forward_prefill` on first call vs 100th call at each bucketed shape.
   Publish as its own wins entry.

## Deferred / out of scope for this plan

- **Qwen3 (non-hybrid) full-forward graph.** Qwen3-4B's prefill path is
  simpler (all full-attn), and the per-layer overhead is a smaller
  fraction of total time. Revisit after Qwen3.5 lands.
- **Multi-batch prefill capture.** Only capture (B=1, seq_len=bucket)
  for this plan. Batch > 1 prefill is rare at infer's current load
  profiles; add later if guidellm shows it matters.

## Rule

Kernel-launch overhead only shows up as a gap when the model has many
kernels per forward. Qwen3.5's hybrid layout multiplies that overhead:
every linear-attn layer calls ~7 individual kernels (conv1d, ssm_scan,
gated_delta_rule, norm, silu, up, down). sglang folds them into a single
graph replay; we must too. This is the canonical "single-model
specialisation beats generic framework" move.
