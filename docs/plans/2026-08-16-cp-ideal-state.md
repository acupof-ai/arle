# CP ideal state: one mesh, one attention-CP core, engine + training

> Status: accepted 2026-08-16 (ckl: "把 cp 重构成理想态,引擎和训练都得支持").
> T1 accepted 2026-08-16 (083e2e89a; gates in
> `docs/experience/wins/2026-08-16-cp-t1-ring-core-extraction.md`). T2 accepted
> 2026-08-16 (gates in
> `docs/experience/wins/2026-08-16-cp-t2b-replicated-kv-prefill.md`; GDR smem
> race found and fixed en route,
> `docs/experience/errors/2026-08-16-gdr-prefill-smem-race.md`). T3 next;
> T2/T3 validation gates listed per tranche.
> T3 revised 2026-08-16 after the engine architecture audit: KV ownership
> sharding priced as the one seam-level change in the 5D program; CP×quant-KV
> and CP×spec combination debts made explicit; position vs DP routing stated.
> T3.1 (B2) implemented 2026-08-17 (807e6c0b4): load-time weight subset
> (W8A16/Marlin preserved), full-head pool at natural head offset, two-buffer
> GDN, global all-reduce, eager (decode graph off at world>1). Pod gate pending.

## Current state (grounded)

| Piece | Where | Status |
|---|---|---|
| Mesh (tp/pp/ep/attn_dp/attn_cp/moe_dp + rank coords) | `infer-topo/src/topology.rs:17-346` | One source of truth; training's `CpContext` is a view over it (`train/src/context_parallel.rs:9-12`) |
| Training full-attn CP (ring, flash-2 merge, zigzag positions-as-data) | `autograd/src/ops/ring_attention.rs` (850 L), `autograd/src/backend_cuda/ring_attn.rs` (728 L, FA3 glue + NCCL ring) | Verified: 27B cp=2 seq=131072, backward peak 94,175 MiB (2026-08-02/03 wins) |
| Training linear-attn CP (a2a sequence→head axis) | `autograd/src/ops/linear_attention.rs:366` | Verified same runs; needs `value_heads % cp == 0` (27B: 48) |
| Training launcher | CLI `cp_size`/`dp_size` (`args.rs:2557`), env `ARLE_TRAIN_{DP,CP}_SIZE` | world = cp×dp, weights replicated (no TP) |
| Engine CP prefill (replicated-KV) | `executor/qwen35.rs:1124-1180`, `qwen35_attention.rs:1394-1479` (GDN relay) | **T2 accepted 2026-08-16** (1.75× at 128K); prefill-only, every cp rank's pool covers the whole prefix |
| Engine CP decode | — | Not built (T3) |
| Engine CP combination matrix | `executor/qwen35.rs:643-644,1025-1029,1989` | Hard mutex today: quant-KV (attn_cp>1 forces bf16), `--kv-recall`, all spec paths (`cp: None`); decode unimplemented |

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

### 3. Engine CP decode (T3) — phased: B2 head-sharding first, then KV ownership

T3 is a phased program. **T3.1 (B2) is self-contained and ships the 256K gate**;
T3.2+ are the strategic KV-ownership program. B2 keeps KV replicated, so it does
not depend on T3.2 and lands first.

#### T3.1 — B2: CP decode = head-sharding across the cp group (ships 256K gate)

> As built 2026-08-17 (807e6c0b4). Decode is **weight-bandwidth-bound**, not
> compute-bound (marlin W8A16 GEMMs ≈52% of the 27B decode step; FA3 chain
> <4%), so the win is sharding the attention *weights*, not the compute. The
> v1 "per-step slice" design was rejected: `slice_rows` dequantizes W8A16 to
> DenseBf16 (loses Marlin) and a per-step D2D copy costs 1.5× the weight traffic
> it saves at cp=2. The as-built design loads a second, finer-sharded weight
> copy once at model load.

The decode regression is not KV capacity. 27B has 16 full-attn layers × 4
kv_heads × head_dim 256, so 256K paged KV = 16×4×256×2×2B = **17 GB** — fits one
H20. The regression is that cp=2 cannibalizes attn_tp (world=2: attn_tp 2→1), so
every rank holds ALL attention heads and doubles qkv/o_proj weight read, KV read,
attention compute, and GDN.

B2: under CP decode the cp group acts as additional attn_tp ranks. Each rank
computes 1/(attn_tp×cp) of the attention stack's heads; the partial hidden
all-reduces over the **global** comm (attn_dp=1 under CP, so attn_tp×cp==world).
Mathematically identical to attn_tp=world decode — recovers the full regression.

- Engage: `cp>1 && dspark off && kv_seq_len+1 >= 8192`
  (`CP_DECODE_MIN_KV_TOKENS`, the cp all-reduce amortization floor). Below
  threshold decode runs replicated — the 4K wash gate.
- Head subset: q subset = attn_tp shard ÷ cp (`local_q_heads/cp`, offset
  `cp_rank × local_q_heads/cp`); kv subset = q ÷ GQA. Guards: q/kv/linear-k/
  linear-v head counts all divisible by cp.
- **Weights (load-time subset, W8A16/Marlin preserved):** a second
  `FullAttnDecode`/`LinearAttnDecode` weight copy at 1/(attn_tp×cp) heads, loaded
  once via the quant-aware sharded load (`load_matrices_row_fused` for qkv rows,
  `load_matrix_sharded_quant_aware(Row)` for o_proj cols). Zero per-step slicing.
  `decode_attn_cfg = TpConfig{world: attn_tp×cp, rank: attn_tp_rank×cp +
  attn_cp_rank}` — a fresh head-shard enumeration (bijection onto the mesh).
- **KV pool (full-head, subset at natural offset):** the pool keeps the full-head
  layout — prefill all-gathers KV so every rank's history covers the whole
  prefix. The decode subset reads/writes its head block at offset
  `cp_rank×decode_kv` (pointer arithmetic on the pool base, no kernel change, no
  migration). GQA maps compact Q head `qh` to compact KV head `qh//GQA`; the
  offset translates it to the correct absolute KV head (`q_off = GQA×kv_off`).
  New-token pages hold only this rank's subset (the other heads' slots are stale
  but unread under B2).
- **GDN decode (two-buffer):** the full pair (all heads) serves prefill and
  non-B2 decode; a 1/cp-head `gdr_states_decode`/`conv_states_decode` pair is
  scattered from the full pair on the first B2 step (idempotent per layer) and
  is the live state B2 advances. `decode_recurrent_live` marks which pair is
  live. The decode pair is fresh-allocated (the executor's recurrent free-list
  blocks are full-dim).
- **All-reduce: global `all_reduce_sum`**, NOT `attn_all_reduce_sum` (the latter
  routes to attn_tp-only under cp>1 — the CP-prefill semantic, wrong for decode
  where the reduce must span the cp group too). FFN all-reduce stays global.
- **Eager, not graph:** the decode graph is disabled at world>1
  (`decode_graph_armed = qwen35_decode_graph() && tp.is_single()`), so B2 decode
  runs eager. (The plan's "graph-compatible" gate was wrong for world>1.)
- **Sidecar/restore:** `save_recurrent_sidecar` skips the fresh snapshot on a
  B2-live slot (the full pair is frozen at the scatter point); `restore`
  refuses a 1/cp decode-pair snapshot (dim mismatch → Err → engine full
  recompute) rather than routing it into the decode pair, which would let the
  next tail-prefill advance the frozen full pair. A B2-live slot cannot
  tail-prefill in-place today (no Decoding→Prefilling transition; multi-turn
  re-enters via prefix-attach, which restores a full-pair sidecar).
- Gate: needle ×3 cp=2 vs cp=1 (short prompts = non-B2 wash; prompts ≥8192 =
  B2 engaged); 4K decode wash; 128K decode recover 43→~60 tok/s at world=2;
  256K decode win.

#### T3.2 — 2D KV ownership sharding (attn_tp × cp), world ≥ 4

> Decided 2026-08-17 (ckl: "2D 分片（world≥4）"). Sequence-sharded KV with
> attn_tp × cp 2D sharding. world=2 keeps B2 (T3.1) — the 2D path needs
> attn_tp ≥ 2 AND cp ≥ 2, so world ≥ 4. No decode regression: weights are
> attn_tp-sharded, KV is cp-sequence-sharded, both sharded at world ≥ 4.
> Rejected: head-shard pool (ties KV to head count, no prefill speedup);
> T3.2-replaces-B2 (flash-decoding at world=2 regresses decode ~20-25% —
> decode is weight-bound, marlin ≈52%, flash-decoding doesn't shard weights);
> B2+T3.2 coexist (two pool layouts, half-state).

**Layout.** Rank (t, c) holds attn_tp shard t's heads for cp sequence shard c:
- Full-attention KV: attn_tp head-sharded + cp sequence-sharded (capacity win, 1/cp per rank).
- GDN recurrent state: attn_tp head-sharded, cp replicated (T2 relay unchanged — state is KBs/layer, recurrence is full-sequence so it can't be sequence-sharded).
- Weights: attn_tp head-sharded (unchanged).

**Why 2D (not the v1 flash-decoding merge).** The v1 decode merge
(flash-decoding, all heads per rank) loses B2's weight-sharding: at world=2
(attn_tp=1, cp=2) each rank reads ALL heads' weights — 2× B2's weight read,
~20-25% decode regression. 2D keeps weight-sharding (attn_tp) AND adds
KV-sharding (cp). At world ≥ 4 both shard; at world=2 the 2D path is
unavailable (attn_tp·cp needs world ≥ 4) so B2 stands.

**Phases.**

T3.2a — prefix matched-length min-reduce (independent prerequisite).
A live divergence window under tier pressure even without CP: rank A and rank B
match different prefix lengths (promote-alloc/mget failure, promote_block
refusal, rank-local top-up eviction, attach_pages failure, sidecar miss —
prefix.rs:761-771,799-816,839-854,138-144,150-162,177-195), then
`prefill_start_pos` diverges (prefix.rs:238) and the planner emits mismatched
rows → TP collectives desync (lib.rs:933). Fix: cross-rank min-reduce of the
restored length via the existing `BackendExecutor::tp_sync_min`
(infer-seam/lib.rs:331; sole caller lib.rs:1523), then truncate the slot and
clamp `prefill_start_pos` to the reduced minimum before `build_forward_plan`
(lib.rs:906). Files: infer-core/src/prefix.rs, infer-core/src/lib.rs.

T3.2b — 2D sharding (big-bang; pool sharding breaks prefill and decode
simultaneously, so B–E land together behind the world≥4 2D mode).

Prefill decision (2026-08-17, ckl "A 单次 ring pass"): under 2D the prefill is
a **single ring pass over the whole prompt** (Megatron model) — chunked
prefill is abandoned in the 2D regime. The 256K capacity SLO is decode-bound
(prefill KV is transient), so the ring pass's loss of decode-interleaving /
preemption / chunked-TTFT for 2D requests is accepted. Industry survey
(vLLM #26133/#46358, SGLang #21637, Megatron CP): sequence sharding and the
decode merge are precedented; **chunked-prefill-under-CP is unsolved
everywhere** (vLLM deferred, Megatron N/A), which is what the single-pass
decision avoids. Two industry gaps are ARLE differentiators: prefix cache
under CP (phase C) and hybrid GDN under CP (vLLM Phase 3).

- B. Pool sequence-sharding. `TokenKVPool` (cuda-kernels/paged_kv.rs:57) keeps
  per-rank bare-u32 identity, but under 2D rank (t,c) allocates only shard c's
  pages. `KvPrefixStore::attach_pages` (infer-seam/prefix_store.rs:10) attaches
  only the local shard's block ids.
- C. Radix replicated across cp + location table. The radix match is from the
  start of the prefix, so a per-rank partial radix cannot match (rank c has no
  block 0). The radix is instead replicated across cp: every cp rank holds the
  full block→page mapping with a location table (block B → shard c(B), page_id
  on rank c(B)), built by a collective exchange (each rank broadcasts its
  shard's blocks+page_ids). Residency is per-shard (block B resident iff rank
  c(B)'s page is live); the T3.2a min-reduce aligns the matched length across
  cp ranks — a missing block on any shard truncates the match for all.
  `publish_prefix_blocks` (prefix.rs:294) publishes the local shard's pages.
  **Prefill reuse is disabled under 2D initially** (the ring pass recomputes
  the whole prompt — gathering the paged cached-prefix shard into the ring
  block is a follow-up); the location table serves decode reuse (multi-turn
  attach) first.
- D. Ring prefill (single pass). Replace `cp_share_chunk_kv`
  (qwen35_attention.rs:1088, replicated gather) with one ring pass over the
  whole prompt: rank c computes its q-shard (prompt/cp) and KV shard, ring-
  rotates the KV shards (P2P), attends to all shards via T1's ring core
  (cuda-kernels/ring_attention.rs: `ring_forward_tile`, `ring_block_fwd_merge_fa3`),
  writes only its shard's KV. No chunking under 2D. GDN relay unchanged
  (qwen35_attention.rs:1341) — the recurrent state is cp-replicated, relayed
  alongside the ring.
- E. Flash-decoding decode merge. Decode: rank (t,c) computes shard t's heads'
  partial attention over shard c (FA3 split-KV over the local shard), merges
  the (m,l,out) partials across cp (one collective per layer on `comm_stream`),
  then all-reduces across attn_tp (global). SGLang #21637 validates the shape:
  pack (output + fp32 LSE), one collective per layer, then a **separate local
  merge kernel** (not fused) — ARLE's merge math is T1's `merge_block`.
  Collective = all-gather the cp partials (small: [batch·heads, head_dim]) then
  a local device merge kernel (new — FA3's split-KV merge is fused in-kernel;
  host `merge_block` is U2-gate-only). On `comm_stream` (tensor.rs:182, unused
  by collectives today) with compute↔comm fences (tensor.rs:637). GDN decode:
  attn_tp-sharded, cp-replicated (redundant across cp, but GDN compute is small).

Files: cuda-kernels/src/{paged_kv,ring_attention}.rs,
infer-seam/src/{kv_query,prefix_store,lib}.rs,
infer-core/src/{radix,prefix,lib}.rs,
infer-cuda/src/{tp,qwen35_attention,qwen35_workspace,qwen35_forward}.rs,
infer-cuda/src/executor/qwen35.rs, + one new merge kernel (cuda-kernels).

Gate: needle ladder ×3 at world=4 (attn_tp=2, cp=2) vs the world=2 B2 /
world=4 attn_tp=4 envelope; 256K capacity at world=4; decode no-regression vs
B2; dated wins entry.

#### T3.4 — Combination debts T2 guarded rather than solved

- **CP × quant-KV**: `attn_cp>1` forces bf16 today (`executor/qwen35.rs:643-644`).
  256K is the regime where quantized KV matters most; T3 must decide whether
  the shard format carries quantized KV (per-rank scales/norms planes,
  `cuda-kernels/src/paged_kv.rs:1080`) or the mutex stays.
- **CP × spec**: every spec path hardcodes `cp: None`
  (`executor/qwen35.rs:1989` et al.) and the verify signature has no CP
  parameter. Decode-time spec under CP needs cross-rank accepted-length
  negotiation — no rank sees all positions; the DSv4 `tp_lockstep_accept`
  broadcast (`dsv4/dspark.rs:84-119`) is the template.
- **CP × recall**: `--kv-recall` is rejected with `attn_cp>1`
  (`executor/qwen35.rs:1025-1029`) because rank-local recall scores diverge the
  cp group's collective schedule.

T3.1 engage by threshold: `cp_decode_min_kv_tokens` (default 8192). Short
requests decode as today (replicated heads, no cp reduce) — no cross-rank cost.
T3.1 gate: decode tok/s at 4K (wash) and 256K KV (win), plus needle ladder ×3.
T3.2+ gates are set with their own designs.

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
- SP as an independent axis (reduce-scatter exists in the collective trait,
  zero callers; decode is memory-bound, not activation-comm-bound — revisit on
  a measured signature).
- CP for the diffusion executor family.

## Position in the 5D sequence (2026-08-16 architecture audit)

- TP is complete on the engine side (example-level on the training side). CP —
  this plan — is the main 256K SLO axis. DP routing (N engine instances + a
  router above, in infer-server) is the first *new* structure for train/infer
  unification: the router does not exist today despite `architecture.md:163`
  claiming DP is clean. It neither blocks nor is blocked by this plan.
- PP and SP stay non-goals: neither the 31.2 GB target model nor any measured
  decode signature justifies their cost.
- Prerequisites that touch CP and should land before or with T3, tracked on
  their own tickets: prefix-match min-reduce (live divergence window, see T3),
  hot-path collectives onto `comm_stream`, DeepEP per-layer host stalls
  (`deepep.rs:376,449,599,659` — four sync points per MoE layer).

## Tranche order and gates

| Tranche | Content | Acceptance |
|---|---|---|
| T1 | Extract shared ring core into `cuda-kernels`; autograd calls it | CP gate battery byte-stable; `cargo test --workspace`; no new public surface beyond the core fns |
| T2 | Engine prefill CP (replicated KV) + GDN state relay | needle ×3 cp=2 vs cp=1; 128K prefill ≥1.6× — **accepted 2026-08-16** (1.75×) |
| T3.1 | B2: CP decode head-sharding across cp group (load-time weight subset, full-head pool at natural offset) | needle ×3 cp=2 vs cp=1 (wash + B2-engaged); 4K decode wash; 128K decode recover 43→~60 at world=2; 256K decode win — **implemented 2026-08-17 (807e6c0b4), pod gate pending** |
| T3.2 | KV ownership sharding (3 seam-level assumptions, §3.2) + decode merge on comm_stream (§3.3) | capacity past 512K; 256K quant-KV; dated wins entry |
| T3.4 | Settle CP×quant-KV, CP×spec, CP×recall (§3.4) | per-debt gates |
| T4 | Cleanup (scalar kernel), GDN a2a scaling decision, cp=4 256K curve; GDR smem-race fix (1f7948070) measured-after pod re-gate before T3 baselines | dated wins/errors entries |
