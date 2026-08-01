# Where a 27B step actually goes: prefill is a serial scan, decode is a scalar GEMV — 2026-08-01

> Status: Characterization. No runtime change; this is the map every later
> optimization is ranked against.

## Context

Three consecutive draft-attention rewrites failed to transfer to the serve
([reduction axis](../errors/2026-08-01-draft-attention-reduction-axis-was-not-the-cost.md),
[IDIV](../errors/2026-08-01-draft-attention-idiv-win-is-microbench-only.md)).
Rather than attempt a fourth, profile the whole step and rank by share.

Two `nsys` captures, GPU-idle H20, ThinkingCap-Qwen3.6-27B-FP8 (dense, 64 layers
= 16 full-attn + 48 linear-attn, 31.2 GB).

## Decode — 25 ms/step, 1094 launches

Plain decode, no spec, 59 steps.

| kernel | launches/step | ms | share |
|---|---:|---:|---:|
| `fp8_gemv_batch_kernel` | 400 | 13.8 | 66% |
| `gemv_handwritten_kernel` (bf16) | 97 | 4.3 | 21% |
| `gated_delta_rule_decode` | 48 | 0.80 | 4% |
| rms_norm / add / silu | ~250 | 0.79 | 4% |
| flash attn | 16 | 0.20 | 1% |
| GPU idle between launches | — | ~4 | 16% |

H20's achievable read is **3.5 TB/s** (measured 2026-07-10), not the 4.02 spec
sheet, so 31.2 GB of weights floors a step at 8.9 ms. The GEMVs take 18.1 ms —
**49% of achievable**, independently reproducing the 51% already attributed to
the per-row activation load+convert tail, where two informed fixes were killed.

The DSpark draft attention this week's work targeted is **1.5 ms of a 35 ms
step — 4.3%**. Its −33.2% microbench win was Amdahl-capped at −1.4% before it
was written.

Verify decomposes as **22 ms intercept + 2.48 ms/row** (5.18 ms/row at 33K).
The intercept is batch- and context-independent and equals one plain non-spec
step: verifying 8 speculative tokens costs what decoding 1 costs. Spec decode
is doing its job; the intercept is the wall.

## Prefill — 33K in 28.6 s

Single request, 24.0 s GPU-busy, ~37K launches, and **2328 `cuMemcpyDtoH`
costing 1.58 s** — host round-trips inside the prefill loop.

| kernel | launches | s | share |
|---|---:|---:|---:|
| `gated_delta_rule_prefill_recurrent` | 1152 | **9.37** | **33%** |
| DeepGEMM FP8, all shapes | 7936 | 8.33 | 29% |
| TileLang full attention | 368 | 3.93 | 14% |
| `pack_quantize` bf16→fp8 | 9600 | 1.50 | 5% |
| conv1d / norm / silu | 3840 | 0.55 | 2% |
| GPU idle (includes host tokenization) | — | ≤4.6 | ≤16% |

Each part against its own ceiling:

- **DeepGEMM is healthy.** `gate_up` moves 11.8 TFLOP/layer in 59.2 ms = 199
  TFLOPS; `down` 189 TFLOPS. Against H20's ~296 TFLOPS FP8 peak that is 64–67%.
  The GEMMs are not the problem and do not need work.
- **Full attention runs 214 TFLOP in 3.93 s = 54 TFLOPS**, 36% of the ~148
  TFLOPS BF16 peak. It is on the TileLang kernel, not the FA3 the decode path
  already uses.
- **The linear-attention recurrence is not compute-bound — it is a latency
  chain.** 8.14 ms per (layer, chunk) launch over ~1375 tokens = **5.9 µs per
  token**, ~11800 cycles, spent on ~6 dependent `__syncthreads` and shared
  reductions per token. No free parallel axis is left to take: the block is
  already 512 threads (`val_dim 128 × j_slice 4` splitting `key_dim`) and the
  token axis *is* the recurrence. The `<<<48, GDR_BLOCK_DIM>>>` grid does starve
  a **78-SM** GPU, but only at c=1 — the varlen launcher uses
  `grid(num_value_heads, batch)`, so c=16 gives 768 blocks. Shortening the
  sequential chain is the only lever; widening the grid is not.

## What Worked

Profiling the step before touching a kernel. One capture reordered the entire
backlog: the target three rewrites went after is 4.3% of decode, while a single
kernel nobody had measured at this context length is 33% of prefill.

The prefix cache is also worth its own line — the same 33K prompt twice runs
**35.1 s then 0.525 s**.

Two ready-looking flags are inert and must not be costed into a plan:
`--qwen35-decode-graph` prints `ARMED` and emits zero `cuGraph*` calls;
`--qwen35-gdr-chunked` shape-guards on `local_linear_v_heads == 32` and this
model has 48, so the TileLang chunked kernel baked for Qwen3.5 never fires.
The CUDA chunked path (`csrc/recurrent/gdr_prefill_{batch,solve}.cu`) has an FFI
declaration and no call site.

## Rule

**Rank by share before ranking by idea.** A kernel's cost curve, its source, and
even its own `ncu` counters answer "how is this kernel bound" — none of them
answer "is this kernel worth fixing". Only the whole-step profile does, and it
has to be taken at the context length that costs the time: the same recurrence
was recorded at 28% of prefill on a shorter shape and is 33% at 33K, while the
draft attention that looked like a third of the draft forward is 4% of the step.

**Name the achievable ceiling, per part, in the same table.** "30% MFU overall"
would have hidden that the GEMMs are already at 67% and the loss is entirely in
one serial scan and one unpromoted attention kernel.
