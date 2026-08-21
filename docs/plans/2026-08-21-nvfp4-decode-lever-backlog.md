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

And `ncu` says why it is not taken: Marlin is at 87% of SM peak — *issue*-bound,
not bandwidth-bound. It is spending its instruction slots, not its bytes. That
single fact orders everything below: the levers that reduce **instructions per
weight byte** come first, and the levers that reduce **bytes** only pay once the
kernel is actually bandwidth-limited.

## Ranked

### 1. An sm_90-native mixed-input GEMM to replace Marlin — OPEN, the only 2x-class lever

**Marlin is an sm_80 kernel running unmodified on Hopper.** Its inner loop is
`mma.sync.aligned.m16n8k16` with `cp_async` staging
(`csrc/gemm/marlin/marlin_template.h:86,92,688,784`), and its only architecture
guard is `__CUDA_ARCH__ < 800`. There is no `wgmma` and no TMA anywhere in it.

That is exactly the wrong shape for an issue-bound kernel on sm_90:

| | Marlin (sm_80 path) | sm_90 native |
|---|---|---|
| MMA | `mma.sync` m16n8k16, one per warp | `wgmma` m64nNk16, one per **warpgroup** |
| weight staging | `cp.async`, per-thread address math | TMA, one descriptor per tile |
| pipeline | synchronous warps | producer/consumer warp specialisation |

Each row removes issue slots per unit of math — the quantity `ncu` says is
binding. Nothing about occupancy, tiles or bank conflicts does, which is why
[the three occupancy tunings](../experience/errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md)
all failed.

**The machinery is already in this tree.** The DSv4 W4A8 MoE path vendors
CUTLASS's sm_90 mixed-input warp-specialised collective —
`csrc/moe/w4a8/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp`
plus its builder and four supporting headers. It is a *grouped* GEMM built for
MoE; the dense decode shape needs the same collective with a different tile and
scheduler.

**Upstream reference: vLLM's Machete**, which exists for precisely this reason —
a CUTLASS 3.x mixed-input W4 GEMM written because Marlin's Ampere-era `mma.sync`
core leaves Hopper's wgmma/TMA path unused. Port before invent.

**The tradeoff, named.** `wgmma` has a hard M=64 minimum per warpgroup. At
decode M=1-16 that wastes 75-98% of the MMA lanes. *That is the hypothesis under
test*: wasted math is free if HBM is the real limit, and the 24.5%-of-roofline
number says it is. If a small-M `wgmma` config lands below Marlin, the
conclusion is that decode is issue-bound for a reason other than the MMA core,
and the axis dies with a measurement.

**Do not start by writing a kernel.** The pre-test is a standalone benchmark of
the existing DSv4 W4A8 collective against `marlin_fp4_gemm` at the dense decode
shapes (M in 1,4,8,16; N=34816/5120; K=5120/17408). One binary, no runtime
changes, and it settles the axis before any dispatch work.

### 2. `gdr_decode_batch_kernel` — OPEN, unexamined

13.0% of decode GPU time, 398.7 ms over 9,792 launches, 40.7 µs each. No `ncu`
has ever been run on it. Unknown whether it is near a ceiling or has the kind of
headroom the NVFP4 widen kernel turned out to have (4.0x, from two
divergently-indexed tables).

Cost: one `ncu` run — the cheapest information in this document. Settles whether
it is a lever at all.

### 3. Wave quantisation on the Marlin launch grid — OPEN, bounded

Decode is a tall-skinny GEMM (M=1-16 against N=34816). If `N / tile_n` leaves a
partial wave across the SMs, the tail costs a full wave and the fix is one
constant. `ncu --metrics launch__waves_per_multiprocessor` on the four instances
answers it. Independent of #1 and cheap enough to run in the same session.

Distinct from the occupancy tunings already killed: those raised *blocks per SM*;
this is about whether the *total block count* divides the machine.

### 4. Re-quantise the per-channel FP8 weights to NVFP4 — OPEN, but only after #1

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
