# 2026-04-14 · Bench — single-token decode kernel port (Triton → CUDA C)

## Context

After rebasing the remote L4 tree onto `origin/main`, `cargo build -p infer
--release` failed with undefined symbols `fused_gqa_attention_decode` and
`attention_decode_reduce`. Root cause: `chore(cuda): drop dead Triton
kernels + finish Route-A doc cleanup` (`d3136ba`) deleted two Triton AOT
kernels that were still being called from the live Rust single-token
decode path (`infer::ops::attention::fused_attention_decode_into`,
reached from `qwen3/decode.rs:120` and `glm4/decode.rs:126` on every
single-request step) — see the commit message of `fix(cuda): restore
single-token split-KV decode kernels` (`132bc84`) for the full write-up.

Forward fix in `132bc84`: add two new CUDA C kernels to
`crates/cuda-kernels/csrc/attention/fused_attention.cu`
(`fused_gqa_attention_decode_single_kernel`,
`attention_decode_reduce_single_kernel`) that are `batch_size=1`
specialisations of the existing `fused_gqa_attention_decode_batched_kernel`,
with `current_pos` read on-device from the `decode_meta` buffer to preserve
CUDA-Graph safety. No Rust changes; only `extern "C"` wrappers added.

This file snapshots decode throughput before and after that kernel port,
on the same L4 box (CUDA 13.0, driver 580.82.07, SM 89), same bench
script (`cargo run -r -p infer --bin bench_serving -- request
--prompt-len 512 --output-len 128`), warmup=5, iters=20, bf16,
CUDA Graph on.

## Before (2026-04-14 baseline, commit `d902090`)

From [`project_l4_perf_baseline.md`](../../../../root/.claude/projects/-content-workspace-agent-infer/memory/project_l4_perf_baseline.md).

**Qwen3-4B (8.0 GB bf16), Triton single-token decode + CUDA C prefill:**

| metric               | value       |
|----------------------|-------------|
| Load                 | 3.13 s      |
| TTFT @ 512 tok       | 84.4 ms avg |
| First decode step    | 35.3 ms     |
| Steady TPOT          | 32.75 ms    |
| Decode tok/s         | 30.52       |
| E2E (512+128)        | 4.25 s      |
| Implied HBM BW       | ~244 GB/s (~81% of L4 peak ~300 GB/s) |

**Qwen3.5-4B (9.3 GB bf16), DeltaNet + HD256 FlashInfer decode:**

| metric               | value       |
|----------------------|-------------|
| Load                 | 3.18 s      |
| TTFT @ 512 tok       | 118.3 ms    |
| First decode step    | 41.1 ms     |
| Steady TPOT          | 36.16 ms    |
| Decode tok/s         | 27.63       |
| E2E (512+128)        | 4.72 s      |

## After (2026-04-14, commit `132bc84` on `main`)

Same box, same day, same bench command. Two back-to-back runs for Qwen3-4B
to confirm stability.

**Qwen3-4B, new CUDA C single-token kernel:**

| metric               | run 1       | run 2       | Δ vs baseline |
|----------------------|-------------|-------------|---------------|
| Load                 | 2.88 s      | 2.89 s      | −7.9%         |
| TTFT @ 512 tok       | 88.28 ms    | 88.87 ms    | +4.6 % / +5.3 %|
| First decode step    | 38.20 ms    | 38.21 ms    | **+8.2 %**    |
| Steady TPOT          | 35.90 ms    | 35.91 ms    | **+9.6 %**    |
| Decode tok/s         | 27.84       | 27.83       | **−8.8 %**    |
| E2E (512+128)        | 4.65 s      | 4.65 s      | +9.4 %        |
| Implied HBM BW       | ~222 GB/s (~74% of L4 peak) | | −9 %           |

Run 1 / run 2 delta is < 0.1 % → the regression is real and stable, not
thermal noise. GPU temp pre-bench: 72 C, within L4 nominal.

**Qwen3.5-4B (control — does NOT use the new kernel):**

| metric               | value       | Δ vs baseline |
|----------------------|-------------|---------------|
| Load                 | 3.33 s      | +4.6 %        |
| TTFT @ 512 tok       | 118.91 ms   | +0.5 %        |
| First decode step    | 41.16 ms    | +0.1 %        |
| Steady TPOT          | 36.20 ms    | +0.1 %        |
| Decode tok/s         | 27.59       | −0.1 %        |
| E2E (512+128)        | 4.72 s      | ±0            |

Qwen3.5 is flat-to-baseline on every decode metric, within measurement
noise. This is a clean control: the GPU, driver, CUDA version, scheduler,
and everything else in the pipeline are unchanged. The Qwen3-4B drop is
fully attributable to the Triton → CUDA C swap in
`fused_gqa_attention_decode_single_kernel`.

## What the numbers say

- The new kernel is **correct** (`infer::ops::tests::test_fused_attention_decode_into`
  compares against the CPU reference and passes; 245 unit tests pass end
  to end).
- But it uses only **~74 %** of L4 HBM bandwidth where the Triton kernel
  used **~81 %**, even though the per-thread math is identical.
- The ~7 pp HBM headroom means the port is becoming **compute-bound**
  earlier than Triton — the tile loop is issuing more ALU work per HBM
  byte streamed, or missing latency hiding, so HBM is no longer the
  bottleneck even though we still want it to be.
- Per-step cost grew from 32.75 ms → 35.90 ms = **+3.15 ms / step**.
  That is ~100 bf16 ops worth of FMA budget on an L4 (roughly a full
  extra HEAD_DIM=128 pass), so the gap is squarely in kernel body
  efficiency, not launch overhead (launch count is unchanged — still
  2 launches per layer, decode+reduce, same as Triton).

## Hypotheses for the 9 % gap

Ordered most → least likely. None verified yet.

1. **Triton's software-pipelined tile loop (`num_stages=2`) hides HBM
   latency that my CUDA C port doesn't.** Triton's IR lowers the inner
   K / V tile loop to a double-buffered `cp.async` pattern where the
   next tile is in flight while the current one is computed. The port
   issues synchronous `__bfloat162float` loads inside the QK dot-product
   loop. Re-pipelining the tile load with `__pipeline_memcpy_async` on
   SM 89 should recover most of the gap.
2. **`smem_scratch` reuse pattern forces too many `__syncthreads`.**
   The port reuses `smem_scratch[NUM_WARPS]` across Q RMS, K RMS, per-
   position QK reduction, and split-0 current-token QK. Each reuse
   needs a `__syncthreads`. Counting: ~3 syncs per tile position on the
   hot path, vs Triton which uses warp-local reductions and only syncs
   once per tile boundary. At 64 positions/tile, that is plausibly the
   ~3 ms gap.
3. **Register pressure crossed a spill threshold.** The port keeps
   `q_rot`, `k_rot`, `v_val`, `acc`, `m_i`, `l_i` plus ~4 temporaries in
   registers per thread for the whole kernel. If the live range forced
   a spill to local memory (HBM ~400 cyc), the HBM-BW utilisation would
   *drop* (matching the observed −7 pp) while per-step latency would
   rise. Worth a `ncu --section SpeedOfLight_RooflineChart` pass.
4. **Cold-cache re-load of `cos_cache` / `sin_cache` per thread.** Q
   and K both read the same two bf16 values per `tid` but the port
   does the load twice (once for Q, once for K). Hoisting the cos/sin
   read into a single `__shared__` cache at the top of the kernel
   would trim one HBM round-trip per thread. Small-but-real.

## Next steps

This entry is the **before / after** snapshot CLAUDE.md requires. Closing
the gap is a separate follow-up and should be its own commit with:

1. `ncu` profile of both kernels (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld`,
   `smsp__inst_executed_pipe_fma`, `launch__registers_per_thread`,
   `smsp__warps_launched.avg.per_cycle_active`) to confirm which
   hypothesis is right.
2. Try hypothesis (1) first — `cp.async` + double-buffered tile load —
   because it is the most likely source and requires no algorithmic
   change. If that closes > 5 pp, stop there.
3. If (1) alone is not enough, fold (4) (hoist cos/sin into shared
   memory) as a cheap second pass.
4. Re-run both benches and append a follow-up snapshot under
   `docs/experience/wins/YYYY-MM-DD-bench-single-token-kernel-tuned.md`.

## Rule

> When porting a Triton kernel to hand-written CUDA C, assume
> a **5–10 % latency regression** from lost software-pipelining
> unless you explicitly re-implement `cp.async` double-buffering.
> Budget the ncu pass + pipelining work as part of the port, not
> as a follow-up — otherwise the "forward fix" locks in a real
> regression and blocks future optimizations from being compared
> against a clean baseline.
