# DSv4 decode: systematic path to ~6 ms/token

**Status**: Phase 0/1 measurement pending pod time (queued behind the KV
coverage round). NOTHING below carries an invented gain number — lever order
is decided by measured shares, per the §0.1 cost-after-decomposition rule.

**Gap statement**: measured c=1 decode ≈ 53 tok/s (~18.9 ms/token, TP=8,
support-matrix §0). Target anchor from ckl: **~6 ms/token** (~167 tok/s).
Gap ≈ 3.1×. The anchor itself gets derived, not assumed (Phase 0).

## Phase 0 — theoretical floor (one pod command + arithmetic)

- `cat /host/DeepSeek-V4-Flash-FP8/config.json` → hidden, layers,
  n_routed_experts, num_experts_per_tok, moe_intermediate, kv_lora_rank,
  first_k_dense, vocab. Compute **active-param bytes/token** (FP8 ≈ 1 B/param)
  + KV read bytes/token at the eval ctx (584 B/tok/layer pool + DSA cache).
- Floor(TP=N) = active_bytes / (N × H20 HBM ~4.0 TB/s) + collective latency
  floor (allreduce per layer × rank latency). State the floor for TP=4 and
  TP=8 and WHICH config "6 ms" is achievable in. Cross-check vs SGLang's
  published DSv4 H20 numbers if available.
- Guard rails: `feedback_measured_floor_is_not_physical_floor`,
  `feedback_ideal_roofline_gap_is_not_launch_overhead` — the roofline states
  the FLOOR; it does not attribute the gap.

## Phase 1 — decomposition on the CURRENT stack (single pod session)

The monolith's `ARLE_DSV4_TRACE_LAYER` / operator trace did NOT survive the
rewrite (grep-clean; the stale env row was deleted from environment.md).
Instrumentation-free first, nsys second:

1. **End-to-end ms/token** from `/v1/stats` throughput deltas (no probe bias):
   B=1 steady decode, 2k-token prompt, ≥512 decode tokens. Matrix (one
   variable at a time, same binary/session): {TP=4, TP=8} × {MTP on (default
   D2/T2), MTP off} × {eager, batched lane}. MTP rows report BOTH ms/step and
   ms/committed-token (acceptance folds in).
2. **nsys single-decode capture** (`scripts/_pod_nsys_64k.sh` /
   `_trace_profile.sh` lineage): kernel-time shares over a steady decode
   window — FlashMLA core, DSA indexer, DeepGEMM grouped MoE, dense GEMMs,
   allreduce (count × avg), lm_head GEMM, memcpys. Host-gap share = wall −
   GPU-busy (per `reference_nvtx_range_ending_in_sync_phantom_bottleneck`,
   window framing cross-checked against per-token wall).
3. Sanity anchors from history (hypotheses until re-measured): 2026-06-01
   monolith trace had `attn_swa_all_reduce` avg 9.2 ms/call and
   `ffn_all_reduce` 340 µs/layer-call — if allreduce still owns a similar
   share, it outranks every kernel lever.

## Phase 2 — levers (enumerated, NOT ranked; order = Phase-1 shares)

| Lever | Mechanism | Existing track | License gate |
|---|---|---|---|
| DeepEP-LL / comm | replace per-layer allreduce on the decode path | #61 batched-lane license open | same-binary A/B, ms/token |
| lm_head vocab-shard | 8× weight/rank + logits all-gather | #99 | A/B ms/token |
| MTP always-on + dynamic verify | more committed tokens/step | #89 spec-flip + DSpark C1 (#124) | ms/committed-token + correctness gate |
| DSv4 decode graph | `ARLE_DSV4_DECODE_GRAPH` branch exists (arch doc §0); executor default off | re-license on rewrite | A/B + needle gate |
| Host/lockstep overhead | tick relay + submit path | window fix landed | only if Phase-1 host-gap share is material (B=1 GPU-bound → wash per feedback memory) |
| Kernel fusion (small ops) | — | KILLED twice before (fp8 pair-quantize, swiglu) | paired component A/B only |
| DP-attn | c>1 throughput, NOT B=1 latency | #89 | out of scope for the 6 ms/token anchor |

## Phase 3 — execute top lever(s) by measured share, one at a time,
re-measure after each; every change lands with a wins entry + Δ% row.

## Protocol rules (binding)

Wall-clock per-token framing is the verdict metric; nsys window shares are
attribution only. One variable per experiment. Same-binary same-session A/B.
Spec rows report committed-token rates. Correctness gate (needle ladder)
before any default flip.
