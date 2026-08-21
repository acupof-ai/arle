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

## Ranked

### 1. Re-measure speculative decode — OPEN, highest expected value

The only lever that touches Marlin's 68.3%: acceptance turns one verify into
several committed tokens, so Marlin runs fewer times per output token. At the
35.1% acceptance already measured, call count drops ~26%.

It is currently recorded as a **large net loss** — MTP d=2 at −77%, DSpark
block 6 at −72%
([wins/2026-08-19-nvfp4-marlin-tensorcore.md](../experience/wins/2026-08-19-nvfp4-marlin-tensorcore.md)).
That entry corrects its own attribution: the cause was
`try_fp8_dequant_bf16_gemm_batch` firing at `M >= 2` and re-dequantising all
11.56 G FP8 params per verify (84.35 ms per forward, 84% of the M=3 budget), and
it says in as many words that the numbers stand as *measurements of the defective
build*.

**That defect was fixed, and the whole prefill path has since been rewritten.**
Nobody has re-run it. Acceptance was never the problem — at 35.1% even a perfect
implementation was losing, which is the signature of a per-call cost, not a
hit-rate.

Cost: no code. Three serve configurations on one binary.
Settles it: no-spec / MTP d=2 / MTP d=4 on the 32K chain, same dataset, ITL and
end-to-end.

### 2. Checkpoint quantisation composition — OPEN

Only **54% of parameters are 4-bit** (7.49 GB U8 against 11.56 GB F8_E4M3);
attention, `lm_head` and layers 56-63 are per-channel FP8. This is the only lever
that attacks prefill and decode at once, because it changes how many bytes Marlin
and DeepGEMM read rather than how fast they read them.

Rough shape: an all-4-bit checkpoint would be ~15 GB resident against today's
22.4 GB.

Risk, and it is real: re-quantising already-quantised FP8 to FP4 compounds error,
and the original BF16 weights are not on the box. Must be proven on a few layers
with the eval before any full conversion.

### 3. `gdr_decode_batch_kernel` — OPEN, unexamined

13.0% of decode GPU time, 398.7 ms over 9,792 launches, 40.7 µs each. No `ncu`
has ever been run on it. Unknown whether it is near a ceiling or has the kind of
headroom the NVFP4 widen kernel turned out to have (4.0x, from two
divergently-indexed tables).

Cost: one `ncu` run. Settles whether it is a lever at all.

### 4. Same-base FP4 vs FP8 — OPEN (measurement, not an optimisation)

Every comparison in `baselines.md` and both READMEs is Qwen3.8-27B-NVFP4 against
Qwen3.6-27B-**FP8** — two different models. `Qwen/Qwen3.8-27B-FP8` is now at
`/data00/Qwen3.8-27B-FP8` (29 GB, from ModelScope; the pod has no HF but
ModelScope is reachable). Re-running the chain and the eval against it is the
first honest answer to "what does FP4 buy on this model".

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
