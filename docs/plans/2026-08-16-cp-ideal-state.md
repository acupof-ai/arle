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

Done 2026-08-16 (083e2e89a). The ring merge math ((m, l, out) forward +
backward adjoint), the FA3 pair route, and the per-block FA3 launches moved
into `cuda-kernels/src/ring_attention.rs`, tape-free, on `&Arc<CudaStream>`.
autograd keeps DeviceHandle adapters, f32↔bf16 staging, scalar fallback
kernels, the NCCL ring rotation, and the tape op. Gates green
(`wins/2026-08-16-cp-t1-ring-core-extraction.md`).

### 2. Engine CP prefill (T2) — revised 2026-08-16 after the executor scout

Design correction: the engine's load-bearing invariant is lockstep SPMD —
every rank builds the identical plan and every rank's `PagedKVPool` covers the
whole prefix (`infer-core/src/lib.rs:1790` identical-admissions comment;
prefix-coverage ensures at `executor/qwen35.rs:1085`, `loader.rs` `for_rows`).
Sharding KV *ownership* in T2 breaks that everywhere for no T2 benefit; KV
ownership sharding is T3's problem. T2 shards prefill *compute* and keeps KV
replicated, which needs no ring:

- **All CP logic lives inside the CUDA executor's `submit_prefill_row` path**
  (`executor/qwen35.rs:2572`). Plans, planner, `KvPool`, engine-core stay
  byte-identical across ranks — lockstep is preserved trivially.
- Per prefill chunk, rank r takes its contiguous cp-slice of the chunk's rows,
  runs the full layer stack on `rows/cp` rows, and after computing its slice's
  k/v at each full-attention layer, all-gathers the KV page writes within the
  attn_cp group so every rank's pool again covers the whole prefix. Attention
  reuses `full_attention_paged` (`qwen35_attention.rs:344`) unchanged — a
  q-slice attending a covered prefix is exactly the chunked-prefill shape it
  already handles via `positions`/`start_positions`.
- Causality makes the pipeline natural: with contiguous slices rank r never
  needs rank >r's KV, so rank 0 never stalls and rank r lags only by the state
  chain. Zigzag only if the measured attention imbalance costs >10% wall-clock
  at 128K (contiguous cp=2: rank 1 carries 3/4 of the O(n²) attention work but
  GEMM/GDN work is row-balanced).
- GDN prefill: cross-rank state relay on the existing per-slot recurrent state
  (`Qwen35SlotState.gdr_states`/`conv_states`, `qwen35_state.rs:15`; snapshot
  machinery `qwen35_state.rs:46` already serializes it). Rank r starts layer
  l's GDN after receiving rank r−1's layer-l state (`[heads, d_k, d_v]`, KBs);
  the chain telescopes to shard-stack time + (cp−1) per-layer hops.
- Sampling: only the last-slice rank holds the final row
  (`qwen35_forward.rs:740`); it samples and NCCL-broadcasts the token in the
  cp group so every rank's engine loop stays identical.
- `TpRuntime::from_env_with_nccl` (`tp.rs:172`) appends an `attn_cp` sub-comm
  split after `attn_tp`/`moe_ep` (same fixed collective order on all ranks);
  mesh math already exists (`build_attn_cp_groups`, `topology.rs:335`, zero
  callers today). `attn_cp` divides the tp world per the existing
  `tp % (attn_dp·attn_cp) == 0` rule — world size does not change.
- Known preconditions: RoPE cache default caps `max_seq_len` at 32,768
  (`qwen35_load.rs:577`); the 128K gate config must raise it. DSpark taps and
  sidecar prefix hashing assume whole-prefix visibility — CP prefill is
  mutually exclusive with those features in T2 (guard, don't solve).
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
