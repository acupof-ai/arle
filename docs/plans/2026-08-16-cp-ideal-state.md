# CP ideal state: one mesh, one attention-CP core, engine + training

> Status: accepted 2026-08-16 (ckl: "把 cp 重构成理想态,引擎和训练都得支持").
> T1 accepted 2026-08-16 (083e2e89a; gates in
> `docs/experience/wins/2026-08-16-cp-t1-ring-core-extraction.md`). T2 in
> progress; T2/T3 validation gates listed per tranche.

## Current state (grounded)

| Piece | Where | Status |
|---|---|---|
| Mesh (tp/pp/ep/attn_dp/attn_cp/moe_dp + rank coords) | `infer-topo/src/topology.rs:17-346` | One source of truth; training's `CpContext` is a view over it (`train/src/context_parallel.rs:9-12`) |
| Training full-attn CP (ring, flash-2 merge, zigzag positions-as-data) | `autograd/src/ops/ring_attention.rs` (850 L), `autograd/src/backend_cuda/ring_attn.rs` (728 L, FA3 glue + NCCL ring) | Verified: 27B cp=2 seq=131072, backward peak 94,175 MiB (2026-08-02/03 wins) |
| Training linear-attn CP (a2a sequence→head axis) | `autograd/src/ops/linear_attention.rs:366` | Verified same runs; needs `value_heads % cp == 0` (27B: 48) |
| Training launcher | CLI `cp_size`/`dp_size` (`args.rs:2557`), env `ARLE_TRAIN_{DP,CP}_SIZE` | world = cp×dp, weights replicated (no TP) |
| Engine CP | — | `attn_cp_size` parses in the mesh (`topology.rs:144`) but has **zero consumers** in `infer-cuda`; engine parallelism is TP-only |

## Why the engine needs CP

- ANCHOR is prefill-bound 279:1; a 256K+ prompt's prefill compute and KV do not
  fit one rank's schedule. TP shards heads; it does not shard the sequence, so
  prefill wall-clock and per-rank KV both stay O(seq).
- OPD needs train and engine to agree on one mesh so a pod run declares one
  layout (`tp × attn_cp × dp`) instead of two private conventions.

## Target design

### 1. One attention-CP core, engine- and train-callable (T1)

The ring merge math ((m, l, out) accumulators), the per-block FA3 kernel glue,
and the NCCL ring transport move from `autograd` (train-only) into
`cuda-kernels` as tape-free functions:

- `cuda-kernels/src/ring_attention.rs`: `RingMergeState`, `merge_block`,
  per-block SDPA launch, device ring step (send/recv KV block, GQA-aware).
- `autograd/src/ops/ring_attention.rs` keeps ONLY the tape op
  (`BackwardOp::RingAttention`), its backward, and the host reference math for
  gates — it calls the extracted core.
- No behavior change; the existing CP gates (`cph_parity`, cp=2 seq=32768
  losses, FA3 compounding gate) re-run as the acceptance bar.

### 2. Engine CP prefill (T2)

- `EngineLoadConfig.attn_cp_size` (default 1) → `ParallelTopology` →
  `Qwen35Model`; multiproc serve spawns `tp × attn_cp` workers on the existing
  relay coordinator.
- A long prompt's prefill rows shard across the attn_cp group (contiguous
  blocks; zigzag only if imbalance measures >10%). Each rank runs the shared
  ring core over its shard and writes ITS OWN KV pages — KV ownership is the
  shard map, recorded per request.
- Linear-attn (GDN) prefill under CP is a state relay, not a ring: rank r
  forwards its final recurrence state (`[heads, d, d]`, KBs) to rank r+1.
  Sequential across ranks but each rank's chunk runs at full speed; the state
  hop is negligible vs the chunk compute.
- Gate: needle ladder ×3 at cp=2 vs cp=1 envelope, plus prefill wall-clock on
  a 128K prompt (target ≥1.6× at cp=2 on the FA3 path).

### 3. Engine CP decode over sharded KV (T3)

- Decode q broadcasts to the cp group; each rank computes partial attention
  over its resident KV shard; partials merge with the same (m, l, out)
  recurrence (flash-decoding across ranks: one small collective per layer).
- Must be graph-compatible (decode graph capture is the fast path): the
  collective is a fixed-shape NCCL all-gather of `[b, h, 1, d]+stats`, captured
  in the decode graph like the TP all-reduce already is.
- Engage by threshold: `cp_decode_min_kv_tokens` (default: only when a
  request's KV exceeds one rank's pool share). Short requests decode on their
  prefill-owner rank alone — no cross-rank cost.
- Gate: decode tok/s at 4K (must be a wash — the threshold keeps CP out of the
  short path) and at 256K KV (the win case), plus needle ladder.

### 4. Training composition stays, walls get owners (T4)

- Training keeps ring + a2a; the extraction (T1) removes the duplicate math.
- Known walls, tracked not fixed here: GDN a2a comm grows with cp (candidate:
  chunked recurrence with cross-rank state relay — same primitive as T2's
  prefill relay); head divisibility caps cp at divisors of 16/48; 256K needs
  the cp=4 activation curve measured before any further design.
- Scalar contiguous ring kernel is unreachable on sm_90/head_dim 256 (FA3 is
  the only path): delete it or gate it to the SM/head_dim set it serves.

## Non-goals

- Weight sharding for training (base is frozen + LoRA; replication is correct
  at 27B — revisit only past ~30 GB resident).
- PP anywhere.
- CP for the diffusion executor family.

## Tranche order and gates

| Tranche | Content | Acceptance |
|---|---|---|
| T1 | Extract shared ring core into `cuda-kernels`; autograd calls it | CP gate battery byte-stable; `cargo test --workspace`; no new public surface beyond the core fns |
| T2 | Engine prefill CP + KV shard ownership + GDN state relay | needle ×3 cp=2 vs cp=1; 128K prefill ≥1.6× |
| T3 | Graph-captured decode merge over sharded KV, thresholded | 4K decode wash; 256K decode win; needle ×3 |
| T4 | Cleanup (scalar kernel), GDN a2a scaling decision, cp=4 256K curve | dated wins/errors entries |
