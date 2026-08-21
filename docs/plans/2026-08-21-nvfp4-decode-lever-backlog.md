# NVFP4 optimisation backlog — ranked by measured attack surface, 2026-08-21

> Status: Open. Worked top-down; every row is closed with a measurement, not an
> argument. Runtime `c19b26511` (v0.5.7), Qwen3.8-27B-NVFP4, 1xH20.

## Where the time is

Two SLOs, two different bottlenecks. The 32K agent workload is **154:1 prefill
to decode**, so end-to-end is prefill-dominated while ITL is decode-dominated.

**Decode — GPU kernel time** (`nsys`, NVTX + `cuda_gpu_kern_sum`; the CUDA-event
profile is not usable here, see *Instrument notes*):

| kernel | share | note |
|---|---:|---|
| Marlin, four template instances | **68.3%** | 87% of SM peak per `ncu` |
| `gdr_decode_batch_kernel` | **13.0%** | never examined |
| `rms_norm_batched_offset` | 3.0% | |
| `nvjet_tst_64x8_…splitK` (the 96-row `in_proj_ba`) | 1.6% | |
| `paged_attention_quantized_fa3_partial` | 1.6% | |
| `add_native` | 1.6% | |

`cuStreamSynchronize` is 72.6% of CUDA API time (2182 ms, 214 calls, 10.2 ms
avg), so decode is **GPU-bound, not launch-bound** — 549 ms of `cudaLaunchKernel`
hides inside the wait and is not a lever.

**Prefill — per-op** (`ARLE_QWEN35_QUANT_PROFILE`): `dense_gemm` 89.5%, and it is
DeepGEMM at 93% of this card's FP8 peak. Everything else is ≤5%.

Both dominant kernels are already near their hardware ceiling. **The levers that
remain reduce call count or reduce bytes — not kernel time.**

## The headroom, in one number

Decode reads **20.03 GB of weights per token** (every tensor except the MTP
layer and `embed_tokens`, scale tensors included — counted off the safetensors
headers). H20's HBM3 is 4.0 TB/s, so one token cannot cost less than **5.01 ms**.
Measured c=1 ITL is **20.46 ms**.

**Decode runs at 24.5% of its bandwidth roofline. 4.1x is on the table.**

`ncu` says why it is not taken: Marlin is at 87% of SM peak — *issue*-bound,
not bandwidth-bound. Measured directly at the kernel, it reads 100.27 MB in
0.0629 ms at M=1, which is **1,594 GB/s, 39.8% of the card**. It is spending its
instruction slots, not its bytes.

The first attempt to fix that by issuing fewer, wider instructions failed: the
vendored CUTLASS sm_90 `wgmma`/TMA collective reaches only 952 GB/s at M=1 and
is 0.65x Marlin
([errors/2026-08-21](../experience/errors/2026-08-21-sm90-collective-loses-below-m32.md)).
What the same sweep did find is that **Marlin costs the same at M=8 as at M=1**,
which moves the lever from kernel speed to row count and reorders everything
below.

## Ranked

### 1. More rows per Marlin call — OPEN, now the measured top lever

**Marlin costs the same at M=8 as at M=1**: 0.0629 ms for 1, 4 and 8 rows of
`gate_up [34816, 5120]`, identical to four digits. It first moves at M=16
(1.40x) and grows roughly linearly after
([errors/2026-08-21](../experience/errors/2026-08-21-sm90-collective-loses-below-m32.md)).

So the lever on the 68.3% is not a faster kernel at one row. Rows 2 through 8
are free, and the question is what fills them.

Speculative decode is the mechanism: a verify step of `b` requests by `d` draft
tokens presents `M = b*(d+1)` to the same GEMM. At c=1 with `d=3`, M goes 1 to 4
at zero extra GEMM time. The recorded MTP loss (-77%) was measured on a build
whose `try_fp8_dequant_bf16_gemm_batch` re-dequantised 11.56 G FP8 params per
verify; that defect is fixed and the prefill path has since been rewritten.

Settles it: no-spec / MTP d=2 / MTP d=4 on the 32K chain, one binary, ITL and
end-to-end.

### 2. `gdr_decode_batch_kernel` — CLOSED, register-staged state landed

State slice kept in registers across both passes: bit-exact, 2.24× at B=1,
1.28× at B=16, 1.20× at B=32 at the kernel; end-to-end a wash (≈2 % of a
c=1 token)
([wins/2026-08-21](../experience/wins/2026-08-21-gdr-decode-batch-register-staged-state.md)).
The original finding:

13.0% of decode GPU time, and `ncu` says it is nowhere near a ceiling: 17.5-19.4%
of compute peak, 20.2-22.5% of memory, DRAM at 2%, IPC 0.66, achieved occupancy
30.9% against a theoretical 100% that nothing caps. **61.6% of the stall is Long
Scoreboard** — waiting on global loads — with 83.3% of cycles having no eligible
warp. Bank conflicts are noise.
([wins/2026-08-21](../experience/wins/2026-08-21-gdr-decode-batch-is-latency-bound.md))

ncu replay kept only B=2 resident, so its "grid too small" rule is about the
profiler. The per-warp findings transfer; a B=16 capture is needed before any
edit.

### 3. Swap kernels above M=32 — OPEN, gated on #1 landing

The vendored CUTLASS sm_90 mixed-input collective **loses to Marlin below M=32
and wins above it**: 0.65x at M=1, 0.90x at M=16, then 1.08x at M=32, 1.46x at
M=48, **1.90x at M=64**. Marlin at M=1 achieves 1,594 GB/s against the
collective's 952 GB/s, so `wgmma` and TMA do not help at one row — the axis is
not a replacement for Marlin, it is a second arm above a row floor.

Serving decode runs M=1..16 today, so there is nothing to switch to yet. Spec
decode is what pushes M there: `b=16, d=3` presents M=64, exactly where the
collective is 1.90x. **Wire this only after #1 lands and the row count moves.**

Caveat kept open: the measured arm is the array/grouped variant, which pays
per-group pointer indirection a dense Machete-style kernel would not.

### 4. Wave quantisation on the Marlin launch grid — CLOSED, already solved upstream

Marlin's config search already minimises it. `gptq_marlin.cuh:544` computes
`waves = div_ceil(n_tiles, sms * blocks_per_sm)` and the surrounding search picks
the `(thread_n, thread_k, blocks_per_sm)` triple with the fewest waves, breaking
ties on the larger K tile; the launch is then `blocks = sms * blocks_per_sm`
(`:706`). Both serving shapes already reach a single wave on 78 SMs — at
`thread_n=256`, `gate_up` N=34816 gives 136 tiles which one wave covers at
`blocks_per_sm=2`, and `down` N=5120 gives 20 tiles, under one wave outright.

Nothing to tune. Closed without spending GPU time.

### 5. Re-quantise the per-channel FP8 weights to NVFP4 — OPEN, but only after #1

38.2% of the parameters (10.62 B) are stored per-channel FP8 at one byte each.
NVFP4 would halve them, cutting the per-token read ~5 GB — a 25% lower
bandwidth floor.

**It buys nothing today.** At 24.5% of roofline the kernel is not waiting on
bytes, so removing bytes moves nothing; this is a post-#1 lever by construction.
It also costs a checkpoint rebuild and puts 4-bit error on the attention
weights, which are the sensitive ones. Listed so it is not mistaken for a
current option.

## Measurement debt — not optimisations, but the numbers rest on them

- **Same-base FP4 vs FP8.** Every comparison in `baselines.md` and both READMEs
  is Qwen3.8-27B-NVFP4 against Qwen3.6-27B-**FP8** — two different models.
  `Qwen/Qwen3.8-27B-FP8` is now at `/data00/Qwen3.8-27B-FP8` (29 GB, from
  ModelScope; the pod has no HF, ModelScope is reachable). Re-running the chain
  and the eval against it is the first honest answer to "what does FP4 buy on
  this model".
- **Speculative decode is recorded as a large net loss** (MTP d=2 at -77%,
  DSpark block 6 at -72%,
  [wins/2026-08-19](../experience/wins/2026-08-19-nvfp4-marlin-tensorcore.md)) —
  measured on a build whose `try_fp8_dequant_bf16_gemm_batch` re-dequantised
  11.56 G FP8 params per verify. That defect is fixed and the prefill path has
  since been rewritten; the recorded numbers no longer describe this binary.
- **Checkpoint composition**, counted 2026-08-21: NVFP4 14.97 B (53.9%),
  per-channel FP8 10.62 B (38.2%), BF16 2.18 B (7.9%), total 27.78 B.

## Closed — rejected on measurement

| lever | why | evidence |
|---|---|---|
| Marlin decode kernel tuning | 87% of SM peak; three tunings raised occupancy 20.7→30.7% while throughput fell | [errors/2026-08-19](../experience/errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md) |
| Materialiser bank-conflict padding | conflicts real but not the limiter; the FP4 kernel is issue-bound at 87%, and padding takes its occupancy from 19 to ~7 blocks/SM | `ncu` |
| Fold the column scale into its consumer | `linear_attn` produces 52.1% of the bytes it touches and its consumer is the gated-delta recurrence; SwiGLU + residual cover 20.1% = 0.2% of the step | checkpoint tensor count |
| Stream-overlap the materialisation | 2.5% once the widen dropped 592→146 ms, against a second stream, +178 MB double buffer and a dispatch API change | per-op profile |
| Move the 170 MiB scratch off `thread_local` | the thread-locality is load bearing: two threads sharing one buffer and one stream can have the second widen overwrite the first before its GEMM runs | code |
| Fuse the two `in_proj` GEMMs | already fused at load (`load_matrix_pair_fused`); what remains is FP8 + BF16 and cannot row-fuse | `qwen35_load.rs:448` |
| `in_proj_ba` (96-row GEMM) | 1.6% of decode GPU time; its 12.75 µs NVTX span is launch time, hidden behind a 10.2 ms sync | `cuda_gpu_kern_sum` |
| Fused W4A8 prefill kernel | would buy the last 3.4%, against writing a kernel that has to reach DeepGEMM's 93%-of-peak pipelining | per-op profile |

## Instrument notes — read before quoting any number here

Six wrong conclusions came out of this profile in one day, every one from
quoting a reading whose semantics I had not checked.

- **`ARLE_CUDA_PROFILE` synchronises** (`profile.rs`, `stop.synchronize()`).
  Nested spans therefore charge the outer span for every inner span's sync:
  `linear_attention` read 36% "unattributed" with it and **5.6%** under NVTX.
  `gdr_recurrent` read 45.95 µs/call with it and **4.42** under NVTX.
- **NVTX PushPop ranges are CPU-side.** An async launch returns immediately, so
  the range measures launch time, not kernel time. `in_proj_qkvz_gemm` reads
  6.49 µs/call for a GEMM that must move 84 MB — 12.9 TB/s on a 4 TB/s card. The
  physical impossibility is what caught it; use `cuda_gpu_kern_sum` for GPU time.
- **A control that moves invalidates the run.** A c=16 row was discarded because
  the unchanged FP8 arm moved 30.42 → 106.53 out tok/s between runs.
- **A roofline bounds a bandwidth-bound kernel and says nothing about an
  issue-bound one.** The widen was predicted at 3.4% of its GEMM and measured at
  52%.
