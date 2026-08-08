# `pack_quantize` one warp per quantization block — 3.67× on the kernel, −2.98% anchor wall, CUDA, 2026-08-09

> Status: **accept.** `554173b36`. Correctness gate clean on both arms, anchor
> A/B counterbalanced with non-overlapping ranges on four metrics.
> **The SOTA row does not move** — see Scope.

## Problem

`dsv4_deepgemm_pack_quantize_bf16_to_fp8_kernel` is **7.8% of prefill kernel
time** on the anchor (2205 ms of 28,168), moving 593 GB at **0.27 TB/s — 7.6% of
the 3.5 TB/s achievable**. It converts bf16 activations to FP8 blocks to feed
DeepGEMM; none of it is model work. It was the largest single line in the
[data-prep tail](2026-08-09-anchor-window-partitioned-exactly-prefill-arithmetic-is-finished.md),
which as a whole carries 3940 ms of headroom — 2.2× the entire arithmetic
headroom in prefill.

## Root cause

One 128-thread block per 128-element quantization block, so **one `uint16_t` per
thread**: a shared-memory reduction with two `__syncthreads`, and a second pass
that re-reads the same input to scale it.

**It was never memory-bound.** `ncu` at the traced shapes:

| metric | before | after |
|---|---:|---:|
| duration | 101.4 µs | **27.6 µs (3.67×)** |
| executed instructions | 46.79 M | **11.81 M (3.96× fewer)** |
| SM (compute) throughput | 81.3% | 74.9% |
| **DRAM throughput** | **5.2%** | 19.2% |
| achieved occupancy | 89.8% | 83.7% |
| executed IPC | 3.30 | 3.21 |

The speedup equals the instruction reduction. DRAM at 5.2% while the old kernel
runs — the memory system was idle and the SM was saturated with address
arithmetic, reduction, and synchronization.

## Fix

One **warp** per quantization block, four bf16 per lane via `ushort4`, values
held in registers so `amax` and the scaled write share a single read, and the
reduction stays in `__shfl_xor`. No shared memory, no `__syncthreads`. The
`extern "C"` signature, the `cols % 128 == 0` precondition, and every Rust caller
are unchanged; only the grid is divided by 4.

Microbench at the traced shapes, rows 2048:

```
cols= 5120   before  88.5 us   after  25.1 us   3.53x   mismatches=0
cols=17408   before 319.3 us   after  86.3 us   3.70x   mismatches=0
cols= 6144   before 107.3 us   after  29.6 us   3.62x   mismatches=0
```

Driving the real entry point against a CPU reference: **0 value and 0 scale
mismatches** at 2048×{5120, 17408, 6144} and at the ragged 336×5120 and
54×17408.

## Correctness gate

`lever_gate.sh` default ladder, `GATE_PROFILE=generic`, `RAW=1
TEMPLATE=qwen3_nonthink NEEDLE_MAX_TOKENS=64`, ThinkingCap-27B-FP8, 3 runs/rung,
both arms:

| rung | 115 | 300 | 446 | 2000 | 8000 |
|---|---|---|---|---|---|
| BASE | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET |
| NEW | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET |

`GATE_EXIT=0` on both, every run emitting the identical needle. The short rungs
that straddle the 241-token boundary are included, and they are the ones this
change could plausibly break through the ragged `max_m`.

## Result — anchor A/B, 32K long-agent, c=16

Dataset sha `8867f63e…` matching `baselines.md`, 128 requests, temp 0, seed
20260416, one fresh serve per trial, order **new base base new new base**, both
models `mlock`ed so every serve reached ready in 10 s.

| metric | BASE mean [min–max] | spread | NEW mean [min–max] | spread | Δ | ranges overlap |
|---|---|---:|---|---:|---:|---|
| wall | 409.16 [408.37–410.27] | 0.5% | 396.97 [394.08–400.89] | 1.7% | **−2.98%** | no |
| total tok/s | 10893.9 [10849.7–10940.3] | 0.8% | 11257.3 [11148.3–11337.8] | 1.7% | **+3.34%** | no |
| TPOT | 271.74 [266.10–277.23] | 4.1% | 257.22 [251.39–261.60] | 4.0% | **−5.34%** | no |
| out tok/s | 39.01 [37.96–40.10] | 5.5% | 41.38 [40.37–42.67] | 5.6% | **+6.07%** | no |
| TTFT p50 | 1882.9 [1858.7–1924.3] | 3.5% | 1913.3 [1866.3–1998.4] | 6.9% | +1.62% | **yes** |
| TTFT p90 | 75794 [72421–77713] | 7.0% | 73595 [73357–73738] | 0.5% | −2.90% | **yes** |

**On wall clock, total tok/s, TPOT and out tok/s the three-trial ranges do not
overlap — the worst NEW trial beats the best BASE trial on each.** Wall clock is
the tightest: BASE occupies 408.37–410.27 s at 0.5% spread against NEW's
394.08–400.89 s. **Both TTFT metrics are inside noise and this is not a TTFT
lever.**

## Scope — the SOTA row does not move

The A/B protocol is **one fresh serve per trial at c=16**; the SOTA row is a
**single serve with an ascending c=1→16 sweep**, where every point above c=1
inherits a warm prefix cache. The two report different quantities — TPOT 257 ms
here against the row's 110.52 ms — so the delta is valid and it is not the row's
cell. Re-measuring the champion needs the ascending sweep on both arms.

**The return is ~55% of prediction and the shortfall is not explained.**
Predicted ~1600 ms from the kernel's 2205 ms share × 3.67×; measured −12.2 s of
409 s. The likely reading is dilution — the anchor's wall includes decode and
queueing where `pack_quantize` is a smaller share than in the pure prefill window
the prediction was priced on — but that is a hypothesis, not a measurement. An
`nsys` capture on the new binary settles whether the 3.67× lands in situ.

## Learnings

**A kernel at 7.6% of bandwidth is not necessarily bandwidth-bound.** The
inference "slow, and it moves a lot of bytes, so it is starved on memory" was
wrong by an order of magnitude: DRAM 5.2%, SM 81%. Low achieved bandwidth is
equally consistent with a kernel too busy to ask for memory. Check SM throughput
and IPC before assigning a bandwidth-class remedy — the four gap buckets have
disjoint remedies precisely so a misassignment is expensive. Concretely: **TMA,
async copy, and warp specialization would have returned nothing here.**

**Predict the end-to-end value from the share before building, then report
against the prediction.** The prediction was made first (~5.4% of wall) and the
measurement came in at 55% of it. Recording the miss is what makes the next
prediction better; a win reported without its prediction teaches nothing.

## Open

- `nsys` on the new binary to confirm the in-situ share and explain the 45% shortfall.
- The rest of the tail is the same pathology: `conv1d` 590 ms at 0.38 TB/s,
  `split2` 360 at 0.80, `rms_norm_gated` 259 at 0.62, `gdr_fq_prep` 207 at 0.51.
- One request on trial 3 (BASE arm) returned `missing output event /
  correctness_error: empty output` after 165 s, 127/128 complete. On the BASE arm
  and inside its envelope, so it does not move the verdict — a dropped serve-side
  stream worth a separate look.
