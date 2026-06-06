# DSv4 prefill — fused wq_a|wkv → FP8 DeepGEMM: −5.05% TTFT (the fusible part of the 30.2% lever)

**Date:** 2026-06-06. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** **opt-in flag** `ARLE_DSV4_FP8_LINEAR_DEEPGEMM=1`, default-off,
licensed by a same-binary prefill_ms A/B. The MLA-LoRA projections are the #1
prefill kernel bucket (scalar `dsv4_fp8_gemv_batch_tiled` = 30.2% of 4096-prefill
GPU, [`2026-06-06-dsv4-prefill-profile-levers.md`](../../plans/2026-06-06-dsv4-prefill-profile-levers.md)).

## What worked — and the kill inside it

- **Fused `wq_a|wkv` → ONE FP8 DeepGEMM** (`run_fused_wqkv_prefill`, seq_len>1,
  gated): the proven decode fused-wqkv pattern (+18.4% at decode) lifted to
  multi-token prefill. Pre-allocated per-layer scratch, decode path untouched.
  - **A/B (4096-tok, same binary, env flip):** OFF `prefill_ms=16905.705` → ON
    `16051.117` = **−854.6 ms, −5.05% TTFT**. clean_tokens=[344] both (token-sane,
    first-token match).
- **KILLED: the broad per-projection route** (wq_a/wq_b/wkv/wo each through a dense
  DeepGEMM): scalar 16624.99 → per-proj DeepGEMM 18165.27 ms = **+9% SLOWER**. The
  per-call pack-quantize + launch overhead dominates for the individual projections
  — only the FUSED form (combining two projections into one GEMM) amortizes the
  overhead enough to win. Same lesson as the residual-GEMV fusion analysis.

## Why only −5% (not the full 30.2%)

The 30.2% scalar-GEMV bucket is wq_a + wq_b + wkv + wo. Only `wq_a|wkv` is
profitably fusible into one DeepGEMM; `wq_b` and `wo` individually lose to the
DeepGEMM pack/launch overhead (the +9% kill). So the realized prefill win is the
fusible slice (−5.05%), not the whole bucket. Capturing more would need a wider
fusion (e.g. fold `wo` into a later GEMM) or a lower-overhead small-M FP8 GEMM.

## Verify / gate

- Gate: token-sane (clean_tokens=[344] both) + first-token match — a
  correct-inference gate for an FP8-GEMM swap (DeepGEMM vs scalar float order
  differs on near-ties; NOT byte-identity). The saved needle prompt
  (`dsv4_needle_4096.ids`) is invalid (its scalar verifier already fails to retrieve
  the needle — a degenerate prompt), so first-token sanity was used.
- Pod build PASS (`ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`); local `cuda,no-cuda`
  typecheck clean. Decode path byte-for-byte untouched.

## nsys decomposition (2026-06-06, #36 license-or-kill) — the +9% is an OVERLAP artifact

Profiled the 4096-prefill scalar-default vs `ARLE_DSV4_FP8_LINEAR_DEEPGEMM=1`
(`dsv4_prefill4096_{scalar,dg}_noflash`, FLASHMLA_DECODE=0 to dodge the rebuild's
sm_90-vs-sm_90a NOT_SUPPORTED). Kernel `total ms` is **summed**, not wall-clock:

| | scalar | fused wq_a\|wkv |
|---|---|---|
| `dsv4_fp8_gemv_batch_tiled` | 1888 inst, 23865 ms, **29.46%** | 1200 inst, 17176 ms, **22.87%** |
| fused projection DeepGEMM (N=1536,K=4096) | — | 344 inst, **66 ms**, 0.09% |
| `pack_quantize_bf16_to_fp8` | (MoE) | 1032 inst, 566 ms, 0.75% |
| 4-byte H2D (active_counts, upper bound) | 39560 cp, 37.5 ms GPU / 99 ms API | +8894 cp, +8 ms / +28 ms API |

**Verdict — NOT (a):** the 4-byte active_counts H2D is negligible (~28 ms API delta
over the whole 8-rank run). Fusing wq_a\|wkv removed ~6689 ms of *summed* scalar-GEMV
and added only ~66 ms DeepGEMM + ~66 ms quantize — yet the clean wall-clock A/B was
only **−5.05%**. **That gap is the §0 framing trap: 29.46% is summed kernel time, not
wall-clock — the scalar GEMVs OVERLAP on streams** (81 s summed kernel / 19.9 s wall
≈ 4× overlap), so 6689 ms summed ≈ ~1.7 s wall ≈ the measured ~850 ms after the
DeepGEMM also serializes. This **reconciles the +9% per-projection kill**: 1:1
per-projection DeepGEMM serializes (quantize→GEMM dependency) and **breaks the
stream-overlap** the scalar GEMVs enjoy, so even though each DeepGEMM kernel is ~50×
cheaper than the scalar GEMV it replaces, the wall-clock regresses. The remaining
22.87% bucket (wq_b/wo + compressor/indexer GEMVs) is **overlap-protected** — not a
clean 1:1 capture. Capturing more needs WIDER fusion (blocked: distinct inputs along
the dependency chain) or stream-overlapping the per-projection prepare (the §5.1
multi-stream lever, higher-effort). **#36 prefill bucket CLOSED at the fused slice.**

## Rule

Routing small individual FP8 projections through DeepGEMM LOSES to the pack/launch
overhead (+9%); only FUSING projections into one GEMM wins (−5%). For prefill GEMV
buckets, the lever is FUSION, not a 1:1 scalar→DeepGEMM swap. **And the nsys %-GPU
(summed kernel time) overstates the wall-clock win on overlapped paths — license on
wall-clock, the summed-% is an upper bound** ([[reference_nvtx_range_ending_in_sync_phantom_bottleneck]]
is the NVTX analog; this is the kern_sum analog). Single-shape (4096) win → kept
opt-in; a multi-shape check + the default-flip is a follow-up. (Flag is an env var per
the test-harness shim convention; → CLI `--` flag at the #35 cleanup,
[[feedback_runtime_config_cli_flags_not_env]].)
