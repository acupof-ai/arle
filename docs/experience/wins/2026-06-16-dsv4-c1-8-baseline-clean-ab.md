# DSv4 8×H20 decode c1–8 — CLEAN baseline + compressor-batch A/B (profiling OFF)

Supersedes the c1–8 table in
[[2026-06-16-dsv4-c1-8-baseline-snapshot]], whose numbers were measured with
`ARLE_DSV4_DECODE_PHASE_TIME=1 + ARLE_DSV4_LINEAR_PROFILE=1` (the
`serve_bench_62.sh` launcher) — each step pays a `cudaStreamSynchronize`, which
**understated tok/s ~25–35%** (the "c1=31.9" was the artifact ckl flagged). This
entry re-measures with **profiling OFF** and adds a same-session gate OFF→ON A/B.

## Goal
Fix the contaminated c1–8 baseline and produce the standing clean anchor for the
next lever (compute/comm overlap on the batched path). "baseline" = current-best
all-non-MTP-opts-ON config ([[feedback_baseline_means_current_best_all_opts_on]]).

## Config
.62 (192.168.12.62), 8×H20 97GB, CUDA 12.9, glibc 2.28. Model
`/data01/models/DeepSeek-V4-Flash`. Binary origin/main@`3e3e50e0` + DSv4
(`/data01/arle-build`). deepgemm JIT cache warm (`DG_JIT_CACHE_DIR=/data01/deepgemm-warm`,
[[reference_dsv4_deepgemm_jit_cache_persist_62]]). nccl-cu12 2.27, clang-11 deepgemm-JIT.

- TP=8, `num-slots 64`, `max-total-tokens 4096`, `chunked_prefill_size 64`.
- `MOE_BACKEND=allreduce`, `EXPERT_BACKEND=deepgemm` (native), `INCREMENTAL_KV=1`,
  `FUSED_DISPATCH_PAYLOAD=1`, batched-FlashMLA decode (default-on c≥4), fused-wqkv,
  decode-proj DeepGEMM, CUDA decode graph — all code-default ON.
- **profiling OFF** (NO `DECODE_PHASE_TIME` / `LINEAR_PROFILE`).
- **NO MTP** — the MTP-head deepgemm JIT hangs on .62's forced clang-11 host
  compiler (toolchain artifact, not a decode regression; needs a gcc≥10 build host
  for the real ~53). See the next-steps plan.
- **A/B = same binary, same session, two configs back-to-back**, only flip:
  `ARLE_DSV4_DECODE_COMPRESSOR_BATCH` (OFF `serve_bench_clean.sh` → ON
  `serve_bench_baseline.sh`).

## Params
Non-streaming `/v1/completions`, `max_tokens=128`, `temperature=0`, one ~28-token
prompt, 2 warmup reqs, c ∈ {1,2,4,8} (c concurrent identical, aggregate
wall-clock). Token count from response `usage.completion_tokens`. Driver
`/data01/run_baseline_ab.sh`, sweep `/data01/sweep.py`. **Single sweep per c**
(see caveat).

## Results — clean A/B (profiling OFF)

| c | OFF agg tok/s | ON agg tok/s | Δ ON vs OFF | ON per-req tok/s |
|---|---------------|--------------|-------------|------------------|
| 1 | 43.8 | 44.9 | +2.5% | 44.9 |
| 2 | 44.1 | 44.8 | +1.6% | 22.4 |
| 4 | 44.2 | 69.8 | **+58%** | 17.4 |
| 8 | 74.0 | 77.6 | +4.9% | 9.7 |

**Headline:** clean c=1 is **~44 tok/s** (ON 44.9 / OFF 43.8), NOT the contaminated
31.9. The OFF column **replicates the prior clean session** (plan doc: 43.0 / 45.0 /
45.0 / 74.8) within noise → the clean gate-OFF baseline is solid cross-session.

**Lever (compressor-batch):** biggest marginal win at **c=4 (+58%)** — where the
per-row compressor GEMVs still dominate OFF (44.2) but ON batches them (69.8). At
c=8 both configs already get broad batching (batched-FlashMLA + natural GEMM/MoE
batch), so the lever's marginal gain narrows to +5% (77.6 vs 74.0). Consistent with
"gain ∝ n until other batched paths saturate."

## Problems / caveats
- **Single sweep per c — the c=4 +58% vs c=8 +5% non-monotonicity is not yet
  pinned to a CI.** The direction (lever helps at c≥4) is solid and matches the
  high-c A/B (n=22, c=64: +38%, [[2026-06-16-dsv4-batched-compressor-prepass]]),
  but the exact per-c magnitude needs ≥3 repeats / median before it's enshrined.
  Labeled hypothesis on magnitude, evidence on direction + sign.
- No TTFT/ITL (streaming `/v1/completions` → HTTP 400 on this build; non-streaming
  only). guidellm not installed on .62.
- Sub-linear aggregate scaling persists (per-req 44.9→9.7 from c1→c8): the decode
  step is ∝ n. Next lever = compute/comm overlap on the batched lane.

## Rule
A committed bench number whose launcher had `DECODE_PHASE_TIME`/`LINEAR_PROFILE`
is profiling-contaminated (per-step sync, −25–35%) and is NOT a throughput
baseline — re-measure profiling-OFF before citing. Always state the bench's
profiling state in the config block.
