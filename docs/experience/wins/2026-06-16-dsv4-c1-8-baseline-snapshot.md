# DSv4 8×H20 decode c1–8 baseline snapshot (all default optimizations ON)

> **⚠️ SUPERSEDED — this table is profiling-CONTAMINATED.** It was served via
> `serve_bench_62.sh`, which exports `ARLE_DSV4_DECODE_PHASE_TIME=1` +
> `ARLE_DSV4_LINEAR_PROFILE=1` — each decode step pays a `cudaStreamSynchronize`,
> understating tok/s ~25–35% (the "c1=31.9" was the artifact ckl flagged).
> **Use the clean profiling-OFF re-measure instead:**
> [[2026-06-16-dsv4-c1-8-baseline-clean-ab]] (clean c1≈44, not 31.9). Kept here
> for the record; do NOT cite these numbers as a baseline.

## Goal
Record the standing **baseline = current best config (all default DSv4 decode
optimizations ON)** at low concurrency (c=1..8), so the next change (compute/comm
overlap for the batched path) is measured against a fixed anchor.

## Config (what "all opts on" means here)
Serve `/data01/serve_bench_62.sh` env + `ARLE_DSV4_DECODE_COMPRESSOR_BATCH=1`:
- TP=8 (`INFER_CUDA_DEVICES=0..7`), `num-slots 64`, `max-total-tokens 4096`,
  `chunked_prefill_size 64`.
- `ARLE_DSV4_MOE_BACKEND=allreduce`, `ARLE_DSV4_EXPERT_BACKEND=deepgemm`
  (native, CUDA 12.9), `ARLE_DSV4_INCREMENTAL_KV=1`,
  `ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1`.
- batched FlashMLA decode (default-on at c≥4), MTP, fused-wqkv decode,
  decode-proj DeepGEMM (all code-default on).
- `ARLE_DSV4_DECODE_COMPRESSOR_BATCH=1` = the compressor-GEMV lever (a4239598) +
  per-slot full-flatten (3e3e50e0).

## Env
.62 node (192.168.12.62, `iv-…bbg7`), 8× NVIDIA H20 (97 GB), CUDA 12.9, glibc 2.28
(build host; binary built there per
[[reference_dsv4_pod_build_topology_61_62]]). Model
`/data01/models/DeepSeek-V4-Flash` (DSv4-Flash). Built from
origin/main@`3e3e50e0` + DSv4 source. nccl-cu12 2.27, clang-11 deepgemm-JIT host.

## Params
Non-streaming `/v1/completions`, fixed `max_tokens=128`, `temperature=0`, one
~28-token prompt, c ∈ {1,2,4,8} (c concurrent identical requests, aggregate
wall-clock). Warmup request before the sweep. (guidellm not installed on .62;
streaming `/v1/completions` returns HTTP 400, so non-streaming → no TTFT/ITL this
snapshot; decode-step ITL proxy is the `[decode-phase]` log, see below.)

## Results — c1–8 baseline (all opts ON)

| c | agg out tok/s | per-req tok/s | req latency (128 tok) |
|---|---------------|---------------|-----------------------|
| 1 | 31.9 | 31.9 | 4.01s |
| 2 | 31.9 | 15.9 | 8.03s |
| 4 | 47.6 | 11.9 | 10.72s |
| 8 | 54.7 | 10.1 | 16.74s |

Aggregate throughput is **sub-linear** (31.9 → 54.7 from c=1→8): the decode step
grows with the batch (per-row compute ∝ n), so per-request rate falls (31.9 → 10.1).

**What the opts buy at low c** (same-binary gate OFF→ON, ΔvsOFF): c=1 32.1→31.9
(~0%, lever idle at n=1), c=2 ~0%, c=4 43.3→47.6 (**+10%**), c=8 47.6→54.7
(**+15%**). The opts help progressively with batch (perrow ∝ n; bigger win at
higher n).

**High-concurrency reference** (n=22, c=64, same binary, from the lever/flatten
A/B): decode step 302.6→218.8 ms, decode 72.7→100.5 tok/s (**+38%** all-opts-on
vs gate-off). See
[[2026-06-16-dsv4-batched-compressor-prepass]].

## Problems / caveats
- Sub-linear scaling is the open item: the step is ∝ n (irreducible per-row
  compute — compressor cache-writes + indexer top-k select + the MoE all-reduce).
  The existing `ARLE_DSV4_COMM_OVERLAP` only covers the `seq_len==1` path, NOT the
  batched (n>1) lane — so the MoE all-reduce is not overlapped at concurrency.
  That is the next lever (cross-layer / cross-expert compute hiding the all-reduce).
- TTFT/ITL not captured (streaming rejected; non-streaming only). Next snapshot
  should wire guidellm or an SSE-capable client for TTFT/ITL.

## Learnings
"baseline" here = the all-opts-ON config (the standing reference), not the
gate-OFF pre-optimization number. Low-c (c≤2) sees ~0% from the decode-batch
levers because n is tiny; the levers are a high-concurrency play (gain ∝ n).
