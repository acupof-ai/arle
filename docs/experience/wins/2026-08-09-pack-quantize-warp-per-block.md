# `pack_quantize` — 16 B loads, 5.13× on the kernel, −2.98% anchor wall, CUDA, 2026-08-09

> Status: **accept.** `554173b36` (4 bf16/lane) then `HEAD` (8 bf16/lane +
> packed conversion). Correctness gate clean on both arms, anchor A/B
> counterbalanced with non-overlapping ranges on four metrics.
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

| metric | before | 4 bf16/lane | **8 bf16/lane, shipped** |
|---|---:|---:|---:|
| duration | 100.7 µs | 27.6 µs | **20.6 µs (4.89×)** |
| executed instructions | 46.61 M | 11.81 M | **7.95 M (5.87× fewer)** |
| SM (compute) throughput | 81.5% | 74.7% | 69.2% |
| **DRAM throughput** | **5.3%** | 19.2% | 25.7% |
| achieved occupancy | 90.3% | 83.7% | 84.6% |
| executed IPC | 3.32 | 3.20 | 3.07 |

The speedup tracks the instruction reduction at every step. DRAM at 5.3% while
the old kernel runs — the memory system was idle and the SM was saturated with
address arithmetic, reduction, and synchronization.

## Fix

**16 lanes per quantization block** — two blocks per warp — with eight bf16 per
lane in one 16 B `uint4` load, values held in registers so `amax` and the scaled
write share a single read, `__shfl_xor` at offsets below 16 so each half-warp
reduces its own block, and `__nv_cvt_float2_to_fp8x2` converting two at a time.
No shared memory, no `__syncthreads`. The `extern "C"` signature, the
`cols % 128 == 0` precondition, and every Rust caller are unchanged; only the
grid is divided by 8.

Microbench at the traced shapes, rows 2048, all mismatches 0:

| cols | before | 4/lane | 8/lane | **8/lane + packed cvt** |
|---:|---:|---:|---:|---:|
| 5120 | 87.6 µs | 24.2 (3.62×) | 18.4 (4.75×) | **17.6 (4.98×)** |
| 17408 | 317.4 µs | 85.4 (3.71×) | 63.2 (5.02×) | **61.9 (5.13×)** |
| 6144 | 106.8 µs | 29.1 (3.68×) | 22.1 (4.83×) | **21.1 (5.06×)** |

**Multiplying by a reciprocal instead of dividing is a further 1.43× (7.2–7.4×
total, 2.5–2.6 TB/s) and was rejected.** It is not bit-identical: against
full-mantissa random bf16 it shifts **3987 / 13335 / 4974 elements** — 3.8e-4 —
by one e4m3 ulp. The extra 1.43× is worth ~130 ms, 0.44% of anchor wall, and
costs the strongest gate available on this kernel. **A structured test pattern
reported 0 mismatches for the same code**; only full-mantissa random data
exposed it.

Driving the real entry point against a CPU reference — **0 value, 0 scale, 0
spill across eight shapes**, for both the 4/lane and the shipped 8/lane form:

| max_m | cols | active_count | |
|---:|---:|---:|---|
| 2048 | 5120 / 17408 / 6144 | 1 | the traced prefill shapes |
| 336, 54 | 5120, 17408 | 1 | the ragged segment shapes |
| 512, 129, 64 | 5120, 17408, 6144 | 4, 3, 8 | **grouped MoE**, ragged per-expert counts, experts in reverse order |

The MoE cases are the real risk surface — the change rewrites the thread→work
mapping that `active_experts` / `active_offsets` / `active_counts` index through,
and the first gate only covered `active_count = 1`. "Spill" fills the output with
a sentinel first and checks that rows past an expert's count are untouched.

## Correctness gate

`lever_gate.sh` default ladder, `GATE_PROFILE=generic`, `RAW=1
TEMPLATE=qwen3_nonthink NEEDLE_MAX_TOKENS=64`, ThinkingCap-27B-FP8, 3 runs/rung,
both arms:

| rung (prompt tok) | 115 (149) | 300 (314) | 446 (453) | 2000 (1989) | 8000 (8129) |
|---|---|---|---|---|---|
| BASE `a37d9cca5` | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET |
| 4/lane `554173b36` | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET |
| **8/lane `5cfe8494f`** | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET | 3/0/0 DET |

`GATE_EXIT=0` on all three, every run emitting the identical `738291`. The short
rungs that straddle the 241-token boundary are included, and they are the ones
this change could plausibly break through the ragged `max_m`. The 8/lane serve
log confirms the changed kernel was exercised rather than a fallback: `Qwen3.5
FP8 dense DeepGEMM warmed 5 projection shape(s) at M=2048`.

## Result — anchor A/B, 32K long-agent, c=16

**Measured on the 4/lane form (`554173b36`).** The 8/lane upgrade adds 1.39× on
top, which moves `pack_quantize` from ~601 ms to ~430 ms: **~171 ms of a 409 s
wall, 0.42%, below this bench's own 0.5% BASE spread.** No A/B is claimed for it
and none was run — it rests on the microbench, the `ncu` instruction count, and
bit-identity.

Dataset sha `8867f63e…` matching `baselines.md`, 128 requests, temp 0, seed
20260416, one fresh serve per trial, order **new base base new new base**, both
models `mlock`ed so every serve reached ready in 10 s.

| metric | BASE mean [min–max] | spread | NEW mean [min–max] | spread | Δ | ranges overlap |
|---|---|---:|---|---:|---:|---|
| wall | 409.16 [408.37–410.27] | 0.5% | 396.97 [394.08–400.89] | 1.7% | **−2.98%** | no |
| TPOT | 271.74 [266.10–277.23] | 4.1% | 257.22 [251.39–261.60] | 4.0% | **−5.34%** | no |
| TTFT p50 | 1882.9 [1858.7–1924.3] | 3.5% | 1913.3 [1866.3–1998.4] | 6.9% | +1.62% | **yes** |
| TTFT p90 | 75794 [72421–77713] | 7.0% | 73595 [73357–73738] | 0.5% | −2.90% | **yes** |

**On wall clock and TPOT the three-trial ranges do not
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

**The return is ~48% of prediction, and the capture below explains it.**
Predicted ~1600 ms from the kernel's 2205 ms window share; measured −12.2 s of
409 s. Not partial engagement — the kernel delivers 5.12× in situ. The window
over-states prefill shares by ~2×.

## Learnings

**A kernel at 7.6% of bandwidth is not necessarily bandwidth-bound.** The
inference "slow, and it moves a lot of bytes, so it is starved on memory" was
wrong by an order of magnitude: DRAM 5.2%, SM 81%. Low achieved bandwidth is
equally consistent with a kernel too busy to ask for memory. Check SM throughput
and IPC before assigning a bandwidth-class remedy — the four gap buckets have
disjoint remedies precisely so a misassignment is expensive. Concretely: **TMA,
async copy, and warp specialization would have returned nothing here.**

**A test pattern can hide the defect it was written to find.** The reciprocal
variant reported **0 mismatches** against a structured input with 4096 distinct
values and **3.8e-4** against full-mantissa random bf16 — same code, opposite
verdict. A bit-identity claim is only as strong as the input's coverage of the
rounding boundaries. The same lesson landed twice that hour: the MoE gate's first
run failed at `active_count = 3` because the harness mapped all three groups to
one expert id, so the reference was ambiguous — the harness, not the kernel.

**Predict the end-to-end value from the share before building, then report
against the prediction.** The prediction was made first (~5.4% of wall) and the
measurement came in at 55% of it. Recording the miss is what makes the next
prediction better; a win reported without its prediction teaches nothing.

## Resolved 2026-08-09 — `nsys` on the new binary

**The kernel is fully engaged and the microbench transfers exactly.** Same 30 s
steady-state window, same analyzer, `70760bc09` against `5cfe8494f`; wall
29,642 / 29,693 ms, GPU busy 96.5% / 96.4%, kernel 28,601 / 28,611 ms.

| | before | after |
|---|---:|---:|
| launches | 15,631 | 15,931 |
| total ms | 2216.19 | **441.24** |
| share of kernel time | 7.75% | **1.54%** |
| **µs per launch** | **141.78** | **27.70 (5.12×)** |

Every one of the 15,931 launches carries the new `/8` grid — the `gridX`
distribution maps 1:1 onto the baseline's at exactly one eighth, with no launch
anywhere at an undivided grid. **So the shortfall was never partial engagement.**

**It was window placement, and the factor is now quantified.** A term of
run-level share `s` sped up `f` times returns `s(1 - 1/f)` of wall:

```
1 - 1/5.12 = 0.805        2.98% / 0.805  =  s = 3.70% run-level
window share of wall = 7.75% x 96.4% busy = 7.47%
7.47 / 3.70  =  2.02x
```

**The anchor's `nsys` window over-states prefill kernel shares by ~2×**, with the
A/B's own 0.5% BASE spread putting the factor in a 1.7-2.4× band. The mechanism
is inferred, not measured: the capture sits 31-38% into the run with all 16 slots
saturated on prefill, while the ramp and drain are decode-heavy at low batch.

**The corollary is bigger than the calibration.** Roughly half this workload's
wall clock is not set by kernel time at all — 128 requests over 16 slots with
TTFT p90 73-184 s is a scheduling regime. Every remaining kernel lever combined
is worth ~11% of wall; scheduling is worth more and is unmodelled.

Two numbers from that capture that must **not** be used: the "where did the
1775 ms go" per-kernel comparison is cross-tree, not cross-kernel — the baseline
predates the draft-attention change, whose `nonpaged_prefill_attention_kernel`
went 16 args to 18 and 270 launches to 30 — and the two profiled runs' throughput
figures are a cross-day, cross-GPU, profiler-loaded comparison, not a
regression. Only the `pack_quantize` row is comparable across the two captures.

## Open

- The other eight tail kernels are the same code pattern: 2248 ms remaining,
  ~3.9% of wall calibrated.
- The shipped form is still instruction-bound: SM 69.2% against DRAM 25.7%. The
  remaining path is fusion into the producing kernel's epilogue, not more tuning.
- The rest of the tail is the same pathology: `conv1d` 590 ms at 0.38 TB/s,
  `split2` 360 at 0.80, `rms_norm_gated` 259 at 0.62, `gdr_fq_prep` 207 at 0.51.
- One request on trial 3 (BASE arm) returned `missing output event /
  correctness_error: empty output` after 165 s, 127/128 complete. On the BASE arm
  and inside its envelope, so it does not move the verdict — a dropped serve-side
  stream worth a separate look.
