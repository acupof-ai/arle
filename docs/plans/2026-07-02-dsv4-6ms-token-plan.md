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

## Phase-0 arithmetic (config.json, round-7b): latency-dominated CONFIRMED

DSv4-Flash: **N=43 layers**, hidden 4096, 256 experts top-6,
moe_intermediate 2048, vocab 129280, head_dim 512, native MTP depth 1.

- Active bytes/token (FP8): ~6×3×4096×2048/expert-layer + shared + attn
  ≈ ~230 MB/layer × 43 + lm_head 0.53 GB ≈ **~10.5 GB aggregate** →
  TP=8: ~0.33 ms/rank, TP=4: ~0.66 ms at 4 TB/s. Bandwidth floor ≪ 1 ms.
- **Collectives/token = 3 × 43 = 129.** For a 6 ms budget with ~1 ms
  compute, average effective collective cost must be ≲ 40 µs. The measured
  18.9 ms implies t_eff ≈ 130 µs equivalent (collectives + gaps) — Phase-1
  prices the split. NVLink small-message NCCL is typically 10-30 µs, so
  **6 ms is arithmetically reachable** if the collective path is tightened
  (H1-via-vendored-mask or H2) and gaps close.

**Phase-1 first pass (TP=4/EP=4, FP8, 2026-07-02)**: MTP-on row measured —
**25.59 ms/committed-token, 59.87 ms/step, 2.34 tok/step acceptance**,
coherent at 1795-token prompts. Rows 2/4/5 (MTP-off baseline, H3 A/B,
nsys t_collective) **BLOCKED by #138**: the MTP-off eager decode lane goes
invisible above a sharp prompt wall ∈ (123, 162] (straddles
sliding_window=128 / first compressed chunk) while the MTP verify lane is
clean — converges with the owner's probe observation (NaN onset ~pos 128)
and the un-root-caused 2026-06-07 ">=122-token decode garbage" memory.
(#137 resolved separately: that was the FP4/MX checkpoint, deleted;
FP8 lane's SHORT-prompt behavior is clean.) Correctness first: #138 owns
the lane debug; Phase-1 resumes on its fix.

## Code facts (source-verified 2026-07-02; see the inventory read)

Per committed token, MODEL1 B=1 eager decode, default env:

- **Collectives/token = 3·N at TP>1** — per layer: Q head-slab all-gather
  (`attention.rs:2750`), attn O-LoRA all-reduce (`dsv4.rs:5055`), MoE
  all-reduce (`dsv4.rs:5308`). No per-step collectives. At TP=1 all are
  no-ops. The collective-latency floor is therefore
  `3·N × t_collective(TP)` — with N from config.json this is computable
  BEFORE any nsys run, and history (monolith trace: `ffn_all_reduce`
  340 µs/layer-call avg) says this term alone may dominate the 18.9 ms.
- **lm_head is replicated** full-vocab GEMV per rank (`dsv4.rs:6286`,
  `loader.rs:74`) — #99 confirmed in code; no gather on the head.
- **Host: 1 sync + 1 D2H(4 B) + 2 H2D per token** (argmax `ops.rs:467`);
  non-greedy adds a full-vocab D2H.
- **Allocations/token**: ~7 `HiddenStates::uninit` device allocs PER LAYER
  (`dsv4.rs:4917-5413`) + 3 tail `DeviceVec::zeros` + the DSv4 tail uses the
  alloc-per-token sampler (`dsv4.rs:4479` → `ops.rs:432`), not the
  zero-alloc scratched one Qwen3.5 uses. Measurable hypothesis, NOT a
  pre-approved fix (B=1 GPU-bound → overhead-removal wash, per memory).
- **MTP D2/T2 (default-on)**: one committed token ≈ 0.33 backbone forwards
  at full acceptance (1 verify over 3 rows + 2 single-layer draft
  forwards); the 18.9 ms/token measured is ALREADY the MTP-folded rate —
  the plain-decode step cost is higher.
- **Decode graph**: exists (`dsv4.rs:5889`), default off, requires
  allreduce transport + no probe lens + MODEL1; sampling stays outside.
- **B=1 never takes the batched lane** (`executor.rs:2798` early-return).

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

## MEASURED (2026-07-03 nsys, TP=4/EP=4, MTP-on) — hypothesis KILLED, retargeted

The emergent H1/H2 hypothesis ("129 serialized collectives dominate") was the
load-bearing assumption, so it was the one to measure — and it is **FALSE**.
Per decode step (~50.7 ms GPU-busy/rank, 2.34 committed tok/step, 25.6 ms/tok):

- **Collectives = 6.7 % of GPU-busy** (3.40 ms; 129 kernels × 26.34 µs avg,
  structure confirmed 86 AR + 43 AG = 43×3). H1 (drop the Q all-gather) saves
  **0.68 ms = 1.3 %**; H2 (all comm) caps at 6.7 %. **Both demoted** — neither
  moves the 6 ms needle. H1's vendored-kernel patch is NOT worth it.
- **The step is GEMV-bound.** FP8 GEMV stack = **52 %** of GPU-busy
  (`dsv4_fp8_gemv_batch_tiled` 27.7 % + `gemv_handwritten` 14.7 % +
  `dsv4_fp8_gemv_batch` 9.8 %); grouped swiglu/down 13.6 %; `dsv4_mhc_params`
  9.5 %; FlashMLA `sparse_attn_fwd` only 2.9 %.
- **Roofline gap is the real bug**: active bytes/token ~10.5 GB ⇒ TP=4 HBM
  floor ~0.66 ms, but measured ~17 ms/row → the small-batch FP8 GEMVs run at
  **~4 % of HBM bandwidth**. 6 ms/token lives HERE, not in comm or launch.

## Retargeted levers (measured shares)

| Lever | Target | Share | Why |
|---|---|---|---|
| **G1 — small-batch FP8 GEMV efficiency** | the 52 % GEMV stack (`gemv_handwritten`, `dsv4_fp8_gemv_batch*`) | 52 % | hand-rolled warp-per-row w8a16 GEMV at R≤8 runs ~4 % of HBM; a better grouped-GEMV / DeepGEMM-at-small-batch / tensor-core path is the main course. ncu the three kernels first. |
| **G2 — MTP acceptance** (2.34 → higher tok/step) | committed-token cost, kernel-free | linear | deeper MTP / better draft (DSpark C1 #124) divides the 25.6 ms directly; orthogonal to G1. Blocked-adjacent by #140 (MTP crash ~613 ticks). |
| **G3 — `dsv4_mhc_params` 9.5 %** | MODEL1 hyper-connection mixer | 9.5 % | suspiciously large for a param-gen op; is it recomputed per layer when it could be cached? read the HC path. |
| H1/H2 (collectives) | — | 1.3–6.7 % | demoted; revisit only if G1 shrinks the GEMV floor enough that 6.7 % matters. |

Both correctness blockers must clear before G1/G2 A/Bs: **#138** (ctx-129
NaN, eager lane) gates any MTP-off measurement; **#140** (MTP crash ~613
ticks) gates any sustained MTP-on run.

## Phase 2 — levers (enumerated, NOT ranked; order = Phase-1 shares)

| Lever | Mechanism | Existing track | License gate |
|---|---|---|---|
| DeepEP-LL / comm | replace per-layer allreduce on the decode path | #61 batched-lane license open | same-binary A/B, ms/token |
| lm_head vocab-shard | 8× weight/rank + logits all-gather | #99 | A/B ms/token |
| MTP always-on + dynamic verify | more committed tokens/step | #89 spec-flip + DSpark C1 (#124) | ms/committed-token + correctness gate |
| DSv4 decode graph | branch exists, default off | **already RE-KILLED 2026-06-10** (B=1 GPU-bound, wash −1.5%, `errors/2026-06-10-dsv4-wholestep-graph-production-path-wash-rekill.md`) | revisit ONLY if Phase-1 host-gap share is material |
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
