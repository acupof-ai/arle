# TinyLM scale sweep — FFI launch overhead dominates; GPU porting per-op hits a ceiling

## Context

After shipping matmul backward on GPU (`0f3cdf6`, +5% iter/s on
d_model=64), the tempting next lever is "port softmax/rms_norm/silu
backwards too — they're still CPU". Before doing that, measure where
time actually goes. CLAUDE.md memory
`feedback_measure_batching_before_ceiling.md`: count kernel launches
per token first.

## Setup

`train_multi_turn --iters 10 --group-size 4 --turns 2 --seq_len 13
(prompt 4 + obs 3 + agent 3 + ...) --backend metal`, varying
`d_model`/`n_heads`/`d_ff` together (d_ff = 2·d_model, n_heads =
d_model/16). All on Apple M-series Metal, bit-identical builds.

## What the numbers say

| d_model | n_heads | d_ff | wall (s) | iter/s | token/s | wall growth vs ÷2 d | d² growth |
|---------|---------|------|----------|--------|---------|----------------------|-----------|
|  64     |  4      |  128 |   4.05   |  2.47  |  128.5  | —                    | —         |
| 128     |  8      |  256 |   5.24   |  1.91  |   99.3  | 1.30×                | 4×        |
| 256     | 16      |  512 |  12.07   |  0.83  |   43.1  | 2.30×                | 4×        |
| 512     | 32      | 1024 |  35.58   |  0.28  |   14.6  | 2.95×                | 4×        |

Two regimes are visible:

- **d_model ∈ {64, 128}** — doubling d grows wall only 1.3×. Per-op
  FFI launch overhead (and Rust-side tape/optimizer machinery) is the
  dominant cost; on-GPU arithmetic is a rounding error. Matmul FLOPs
  at d=64: ~10·d²·tokens·layers = 10 · 4096 · 52 · 2 = 4.3M FLOPs per
  iter — microseconds even at 10 GFLOPS. Iter wall is 400 ms. Ratio:
  overhead is ≥99% of wall time.
- **d_model ∈ {256, 512}** — wall grows ~2.5–3× per 2× d, i.e.
  roughly d². This is the regime where matmul compute finally
  matters: at d=512, Q/K/V/O + FFN is ~10·d² per token-layer, and
  matmul-backward-on-GPU ships real wins here.

## What this rules OUT for now

**Porting softmax_backward / log_softmax_backward / rms_norm_backward
to Metal is NOT the next tractable perf lever.** Those ops are
O(d·n) or O(d·n²/head) per token — tiny FLOP count next to matmul.
At d_model=512 the *total* softmax-backward FLOPs per iter are
O(groups·seq²·heads) ≈ 4 · 13² · 32 ≈ 22k ops. Single-digit μs on
CPU. GPU-porting adds a mandatory FFI launch per op call, which at
TinyLM scale is comparable to the whole op.

## What this rules IN

- **Device-resident tensors (M5.3)** — eliminate per-op
  Vec<f32>→GPU→Vec<f32> round-trips, so FFI call overhead amortizes
  across the whole training step. This is the next architectural
  lever; cannot be shipped as an additive trait method.
- **Kernel fusion / compiled graph** — merge `rms_norm → linear →
  gelu → linear` style subgraphs so one launch replaces ~4. Large
  refactor; likely gated on device-resident tensors anyway.
- **Qwen-scale empirical bench** (d_model ~2048) — should show
  matmul-backward-on-GPU moving the needle well beyond the +5% seen
  on TinyLM. Pending remote CUDA box and/or a Metal Qwen training
  baseline.

## Rule

When the next perf target is "port op X to GPU", measure whether
the current bottleneck is compute or overhead FIRST. A 2× d_model
sweep that shows wall scaling sublinearly (~1.3×) is a tell that
per-op overhead, not arithmetic, owns the budget. In that regime,
moving another op to GPU adds overhead without cutting the dominant
cost.

Concretely: if wall(2d) / wall(d) ≪ 4 on a transformer workload,
stop porting individual ops and route work into (a) device-resident
tensors or (b) kernel fusion instead.
