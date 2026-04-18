# Metal Qwen3.5-4B-4bit concurrent decode — ceiling characterization + P0 landing

**Date**: 2026-04-18
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA)
**Model**: `mlx-community/Qwen3.5-4B-MLX-4bit` (24 GDR + 8 full-attn layers)
**Commit landed**: `e22aebc` — `feat(qwen35): real batched sampling on Metal packed decode`

## Context

Concurrent bench on `metal_serve` against `/v1/completions` (128 tokens/req,
temp=0, ~40-token prompt) across concurrency 1/2/4/8. Plain packed decode,
no DFlash (DFlash auto-disables at `open.len() >= 2` today).

## Bench

### Pre-fix (before batched sampling)

| C | agg tps | per-req tps | step time |
|---:|---:|---:|---:|
| 1 | 63.6 | 63.6 | 15.7 ms |
| 2 | 117.8 | 59.2 | 17.0 ms |
| 4 | 144.5 | 36.4 | 27.8 ms |
| 8 | 145.6 | 26.9 | 55.0 ms |

Linear fit `t(B) ≈ 4.4 + 6.3·B ms` → ceiling ~158 tps, observed ~145 tps.

### Post-fix (batched sampling)

| C | agg tps | per-req tps | step time | Δ vs pre |
|---:|---:|---:|---:|---:|
| 1 | 68.1 | 68.1 | 14.7 ms | +7.1% |
| 2 | 121.9 | 61.2 | 16.4 ms | +3.5% |
| 4 | 144.3 | 36.3 | 27.7 ms | ±0 |
| 8 | 144.4 | 26.8 | 55.4 ms | ±0 |

Linear fit `t(B) ≈ 3.4 + 6.5·B ms` → **constant term `a` dropped by 1.0 ms**
(sampling kernel launches saved), **linear term `b` is unchanged**.

### Reproduction at commit `5fe8805` (later same day)

| C | agg tps | per-req tps | step time |
|---:|---:|---:|---:|
| 1 | 67.2 | 67.2 | 14.9 ms |
| 2 | 120.7 | 60.6 | 16.6 ms |
| 4 | 143.5 | 36.2 | 27.9 ms |
| 8 | 142.8 | 26.5 | 56.0 ms |

Fit `t(B) ≈ 6.0 + 6.1·B ms`; per-row asymptote ~164 tok/s, observed 143 tok/s
(the 20 tok/s gap is HTTP + scheduler + tokenizer + sample loop overhead).
Matches post-fix within noise — the ceiling is reproducible, not an artifact.

### mlx_lm baseline — upper bound at B=1

`mlx_lm.generate --model mlx-community/Qwen3.5-4B-MLX-4bit --temp 0.0
--max-tokens 128` on the same prompt on the same M4 Max:

```
Generation: 128 tokens, 84.363 tokens-per-sec
```

**84.4 tok/s at B=1** → 11.85 ms/step.
Ours at commit `5fe8805` hits 67.2 tok/s (14.9 ms/step) — **3.0 ms/step
headroom**. Not kernel-bound; something in the server path that mlx_lm
avoids.

### P0b: drop redundant eval per step (commit `de7b687`)

Tracing the single-row hot path (`request_state.rs:2438` `run_cpp_step`
+ `:2907` `decode_token` Qwen3.5 standard branch) found **two synchronous
`eval`s per step**:

1. `eval(&step_outputs)` inside `run_cpp_step` — cache arrays + logits
2. `eval(&[&sampled])` inside `decode_token` — argmax result

mlx_lm only syncs once per step (via `.item()` auto-eval) after
`async_eval`. Fix: swap the inner `eval` for `async_eval`, drop the outer
`eval(&[&sampled])` since `item_i32()` auto-evals the dependency chain.
Same pattern applied to `run_rust_step` and the packed-decode batch path.

| C | pre P0b | post P0b | Δ |
|---:|---:|---:|---:|
| 1 | 67.2 | 70.3 | **+4.6%** |
| 2 | 120.7 | 120.0 | ±0 (noise) |
| 4 | 143.5 | 143.2 | ±0 |
| 8 | 142.8 | 142.0 | ±0 |

c=1 closed ~1/3 of the 14 tok/s gap to mlx_lm. c≥2 unchanged — the
kernel-compute ceiling is unaffected because the saved CPU-GPU round-trip
overlaps with GPU work that's already the critical path at concurrent
batch sizes.

### P0c: cross-step pipelining in decode_token (commit `5593448`)

`metal_generate_qwen35` (single-prompt path) already did mlx_lm-style
double-buffered decode (`qwen35.rs:886-910`). The HTTP scheduler path
via `Qwen35StepDriver::decode_token` did not — each call sampled+blocked
synchronously, leaving GPU idle while the scheduler did bookkeeping.

Fix: add `pending_sampled: Option<MlxArray>` to `Qwen35StepDriver`;
after computing the result token, pre-queue step N+1 using `result` as
input and stash the lazy sampled. Next call's consuming branch blocks
on `pending_sampled.item_i32()` and skips the forward (already done).
Pre-queue gated on `!dflash && cache_len + 2 <= kv_capacity` (so no KV
extend mid-pending). Invalidate `pending_sampled = None` on DFlash
entry, `prefill_tokens`, and `import_prefix_snapshot`.

| C | pre P0c | post P0c | Δ |
|---:|---:|---:|---:|
| 1 | 70.3 | 72.1 | **+2.6%** |
| 2 | 120.0 | 124.3 | **+3.6%** |
| 4 | 143.2 | 146.1 | **+2.0%** |
| 8 | 142.0 | 147.0 | **+3.5%** — breaks 145 ceiling |

Surprise: c=8 broke the prior "GDR compute-bound" ceiling. At compute-
bound regime, the remaining 3-4 tok/s was CPU-side scheduler-idle gaps
between kernel dispatches. Keeping the GPU command queue always-fuller
(one step ahead) tightened kernel back-to-back timing.

Remaining gap to mlx_lm at c=1 (~12 tok/s) must be tokenizer/HTTP/
scheduler overhead outside `decode_token`'s scope — a single-prompt
bench against `metal_generate_qwen35` directly would disambiguate.

### P0d: rate-limit scheduler MLX memory FFI (commit `db729d7`)

Direct `metal_bench` vs mlx_lm on the same M4 Max:

```
./target/release/metal_bench --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --prompt-tokens 20 --generation-tokens 128 --warmup 2 --runs 3
→ 84.2 tok/s mean | 84.4 p50 | 84.7 p99
```

Matches mlx_lm 84.4 — the direct path has zero gap. **All the c=1
HTTP gap is in the scheduler path, not the decode internals.**

Micro-bench of `IncrementalDecoder::step()` (HF tokenizers
`DecodeStream::step`) on the Qwen3.5 tokenizer: **0.6-0.7 us/tok**,
~0.09 ms per 128-token gen. Tokenizer is NOT the bottleneck.

Grepping the scheduler loop: `refresh_runtime_metrics` fires 3 MLX
C++ FFI allocator queries (`mlx_get_active/peak/cache_memory`), and the
`run_metal_scheduler_runtime` loop calls it twice per tick — once before
`scheduler.step()`, once after. Each query goes through the MLX
allocator's internal mutex. At c=1 decode that's 6 cross-FFI
lock-acquire round-trips per token.

Fix: guard `refresh_runtime_metrics` with a 40 ms interval (Prometheus
scrape cadence is seconds; 40 ms is plenty). Admission events still
fire an unconditional refresh so first-scrape latency is unchanged.

| C | pre P0d | post P0d | Δ |
|---:|---:|---:|---:|
| 1 | 72.3 | 74.1 | **+2.5%** |
| 2 | 114.0 | 122.3 | **+7.3%** |
| 4 | 137.6 | 145.8 | **+6.0%** |
| 8 | 143.3 | 145.6 | +1.6% (noise band) |

Biggest gain at c=2-4 where the scheduler tick sits on the critical
path between GPU batch dispatches — at c=1 decode is already dominated
by GPU forward latency (so saving CPU scheduler time only shaves the
small pre-queue gap), and at c=8 the tick is busy enough with
8 requests' per-tick bookkeeping that metrics FFI is a smaller fraction
of it.

Remaining c=1 gap to mlx_lm: 84.4 − 74.1 = 10.3 tok/s (~1.6 ms/step).
Still in scheduler + HTTP path: mpsc delta_tx send, active HashMap
remove+insert per tick, StopChunkProcessor string push, emit_event
through the scheduler's event sink, delta_tx.is_closed probe.

## Root cause of the B-linear term

The old sampling path called `argmax(logits)` (flat, wrong for B>1) so
`qwen35_can_batch_sample` was pinned to `false`, forcing a per-row loop at
`request_state.rs:1578-1591` — B `slice_row` + B `gpu_sample_token` kernel
dispatches per step. Fix: new `gpu_sample_token_batched` using
`argmax_axis(-1)` / `random_categorical(-1)`, plus same-params gate on the
batched path. Equivalence test against the scalar path at B=4 passes
token-for-token (`qwen35.rs::packed_decode_batched_sampling_matches_scalar_sampling_for_b4`).

**But the fix only hit the constant term.** The 6.5 ms/row linear term
survives. Mapped in two explorer sweeps (Rust-side + C++-side). Scalar
fast path (plain packed decode, `has_cache_pos_arr=false`) does NOT enter
the per-row `slice_update` loop at `mlx_qwen35_model.cpp:586-630`
(that's gated on Layer 2 verify). So the linear scaling is not kernel-launch
overhead — it's **compute inside `gated_delta_step`**.

## The true ceiling: GDR recurrent kernel

Our `gated_delta_step` kernel grid is `(32, Dv, B·Hv)` with threadgroup
`(32, 4, 1)` — **identical to the `bstnxbt/dflash-mlx` reference
implementation's kernel** (confirmed via web read). So there's no easy
grid-tuning win. The kernel's work is algorithmically `O(B · Hv · Dv · Dk)`
per layer, and we run it across **24 GDR layers per step**. That work
scales linearly in B and is fundamental to the gated-delta recurrence.

On M4 Max the kernel should have plenty of parallelism (64 thread groups
at B=8, 40 GPU cores), so it's likely not occupancy-limited — more likely
memory-bandwidth or dispatch overhead across the 24 layers. **Needs
Xcode Metal GPU capture to confirm**, which is out of scope for the
Claude Code loop.

## The real next lever: compile GDR+MLP sublayers

`mlx_qwen35_model.cpp:1061-1068`:

```
// NOTE: mx::compile() cannot handle position-dependent KV cache slicing
// (cache_pos changes each step, forcing re-trace). For now, skip JIT
// compilation and run the C++ forward directly. This still eliminates
// most Rust/FFI overhead.
//
// Future: compile individual GDR+MLP sublayers (no position deps) while
// keeping full-attention layers uncompiled.
is_compiled = false;
```

This is the real attack surface. The full-attn sublayers are
position-dependent (they slice KV caches by `cache_pos`), but the 24 GDR
sublayers and all 32 MLP sublayers have no position dependency — their
inputs are just `(hidden, layer_weights, gdr_state)` and their outputs
feed back into `hidden` / `gdr_state`. `mx::compile` with shapeless mode
should fuse each sublayer's ~20 MLX ops into one compiled graph per
sublayer. Estimated impact: fewer Metal kernel launches per step (today
probably 600+; compiled would be ~50-100), likely moves both `a` and `b`.

## Classification of remaining levers

| Lever | Moves constant `a` | Moves linear `b` | Effort | Notes |
|---|---|---|---|---|
| P0 batched sampling | **−1.0 ms** ✓ | no | landed | e22aebc |
| P1 cache `cols` in mask | 0 for uniform-length | no | S | `needs_mask` already short-circuits when all `left_padding == 0`; only bites mixed-length traffic. Skip until mixed-length bench exists. |
| P2 reuse rope_offsets | ~0 | no | S | 32 bytes + `from_slice_i32` is already sub-µs. Replacing with a kernel-dispatched `add` would likely regress. Skip. |
| Compile full forward | 0 | 0 | — | **already happening**: `fast::rms_norm`, `silu`, `swiglu`, `mlp` are `mlx::core::compile(shapeless)` compiled (see `mlx_qwen35_model.cpp:254-333`). Outer layer loop is pure graph construction that MLX lazy-evaluates. |
| GDR kernel-level fusion | no | −? | L | **needs Metal GPU capture data first** — without `ncu`-equivalent numbers we can't tell if kernel is compute/memory/launch bound. |
| Algorithmic GDR change | — | — | XL | out of scope |

## What Worked

- Mapping the cost surface with two targeted explorer sweeps (Rust +
  C++) before touching code. Saved an attempt on the wrong layer (the
  per-row `slice_update` loop turned out to be gated on Layer 2 verify,
  not plain decode).
- Web-checking against `bstnxbt/dflash-mlx` confirmed our GDR kernel
  geometry is canonical. Prevented a speculative re-tune.
- B=4 equivalence test token-for-token against the scalar path gave
  confidence the batched sampler wasn't silently changing outputs.

## Rule

For Qwen3.5-style hybrid linear-attn models on Metal: when concurrent
decode plateaus, **check what fraction of the step time is kernel-launch
overhead vs. kernel compute before optimizing**. Our P0 optimized the
kernel-launch term; moving the plateau requires attacking the
un-compiled-forward (many small kernels) or the GDR kernel itself
(compute-linear in B).
