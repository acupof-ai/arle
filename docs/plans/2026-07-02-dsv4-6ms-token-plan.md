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

## Emergent structure (from the code facts; ordering still a HYPOTHESIS until Phase-1)

The inventory converts the gap from mystery to arithmetic. Sketch (exact
numbers await config.json's N/hidden/vocab):

- **Bandwidth is NOT the wall.** Even at ~20B active params FP8, TP=8 reads
  ~2.5 GB/rank/token ⇒ ~0.6 ms at 4 TB/s; latent-KV reads are tens of MB.
  The roofline floor is sub-millisecond — so ~18 of the 18.9 ms live in
  LATENCY terms: 3·N serialized collectives + launch gaps + the replicated
  work.
- **H1 — drop the Q all-gather (3→2 collectives/layer, −33% collective
  latency).** Today every rank gathers the full Q head slab and runs
  ALL-head FlashMLA (attention compute replicated across ranks; the latent
  KV is replicated anyway). Local-head attention + the EXISTING O all-reduce
  is algebraically identical and deletes one collective per layer. This is
  the cheapest structural cut in the inventory.
  **BLOCKED at TP>1 (2026-07-02) — see the H1 feasibility verdict below.**
- **H2 — comm latency per collective**: allreduce of one hidden vector
  (~14 KB bf16) is pure latency. Candidates: DeepEP-LL decode MoE (#61
  license open), NCCL low-latency algo tuning, TP=4-vs-8 A/B (fewer ranks =
  lower per-collective latency; bandwidth headroom says TP=4 may WIN B=1).
- **H3 — lm_head shard (#99)**: replicated full-vocab GEMV reads
  vocab×hidden bytes per rank per token (~0.5-1 GB, ~0.1-0.25 ms) — sharding
  is a quantified, bounded win, not the main course.
- **H4 — MTP deeper/dynamic verify** (DSpark C1 #124): multiplies committed
  tokens per step; orthogonal to H1-H3.
- H5 (allocs/sync hygiene): 7·N device allocs + alloc-per-token sampler —
  measure first; B=1 GPU-bound history says wash.

Phase-1 measurement now has ONE job: confirm the collective/launch share
(expected dominant) and price t_collective(TP) — then H1/H2 go first.

### H1 feasibility verdict (2026-07-02): BLOCKED at TP>1 — FlashMLA head-count contract

Source-read of the full decode call chain (`try_flashmla_decode_attention`
→ `arle_flashmla_sm90_sparse_decode_fwd` → vendored SM90 sparse-FP8 kernel).
The kernel hard-requires the FULL head count; local-head decode cannot run it:

- `crates/cuda-kernels/vendor/flashmla/csrc/sm90/decode/sparse_fp8/config.h:19`
  — `static_assert(NUM_HEADS == 64 || NUM_HEADS == 128)`; `NUM_M_BLOCKS =
  NUM_HEADS / 64`, `CLUSTER_SIZE = NUM_M_BLOCKS`, `BLOCK_M = 64`. The M tile
  is one 64-head block per CTA (GMMA 64-row atoms `MMA_64x64x16_F32BF16BF16` /
  `MMA_64x256x16`, config.h:113-131); 128 heads = a 2-CTA cluster.
- `…/sparse_fp8/splitkv_mla.cuh:692` — host launcher
  `KU_ASSERT(params.h_q % BLOCK_M == 0)`; grid is
  `dim3(NUM_M_BLOCKS, s_q, num_sm_parts)` with compile-time `NUM_M_BLOCKS`
  (`splitkv_mla.cuh:769-775`).
- Only h64/h128 instantiations exist:
  `…/sparse_fp8/instantiations/{model1,v32}_persistent_h{64,128}.cu`.
- The ARLE shim pre-flights the same set: `h_q != 64 && h_q != 128 →
  cudaErrorInvalidValue` (`crates/cuda-kernels/csrc/misc/
  arle_flashmla_decode_shim.cu:260` fwd, `:108` get_meta).
- ARLE already encodes the constraint: `ensure!(matches!(global_heads, 64 |
  128))` (`crates/infer-cuda/src/attention.rs:2629-2632`). The Q all-gather at
  `attention.rs:2745-2780` (and the batched lane's `gather_q_row`,
  `attention/flashmla.rs:1095`) exists BECAUSE of this kernel contract — both
  lanes share it, so H1 is blocked identically on eager, MTP-verify, and
  batched paths.

DSv4-Flash has `num_attention_heads = 64` (config; spec fixture
`crates/deepseek-spec/src/v4.rs:1230`). Local head slabs are 8 (TP=8), 16
(TP=4), 32 (TP=2) — every TP>1 slicing fails `h_q % 64 == 0`. There is no
runnable-parameter escape: the epilogue DOES mask partial head blocks
(`num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M)`,
`splitkv_mla.cuh:326,428`), but the host assert forbids exercising it.

Realistic options (all deferred; none is a padding hack):

1. **Vendored-kernel patch**: relax `KU_ASSERT(h_q % 64 == 0)` and lean on the
   existing `num_valid_seq_q` masking (Q TMA reads are bounds-checked against
   the tensor-map shape, so the 48/56 out-of-range M rows load zero); fix
   `get_meta`'s `num_sms / (s_q * (h_q/64))` integer division (shim:127) for
   h_q<64. The 64-row wgmma then computes 8 valid + 56 wasted rows — decode
   attention is KV-read-bound so the FLOP waste is plausibly a wash, but the
   patch needs its own correctness license (needle gate) + perf A/B on pod.
2. **Split the KV/topk bands instead of heads**: all ranks keep all 64 heads
   but attend disjoint index bands, merged by an LSE-weighted combine (the
   split-KV combine kernel already merges partials with LSE). Turns the Q
   all-gather into an O(heads) LSE+O exchange — a bigger redesign of the
   indices builder + a new inter-rank combine.
3. **H2 route (no kernel change)**: keep 3 collectives/layer but cut
   t_collective — the one-shot AG/AR path is already default-on; remaining
   levers are TP=4-vs-8 A/B and DeepEP-LL (#61).

Consequence: collectives stay 3·N/token on the FlashMLA decode lane at TP>1;
H1 produces no runtime knob. H3 (lm_head shard, #99) proceeds independently
(`ARLE_DSV4_LM_HEAD_SHARD=1`).

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
