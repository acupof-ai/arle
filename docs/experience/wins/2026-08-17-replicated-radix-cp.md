# Replicated radix across CP shards — engine, 2026-08-17

> Status: pending-remote

## Goal

Keep prefix reuse correct under 2D (attn_tp × cp) KV sequence-sharding
(T3.2b Part C). Each cp rank's pool holds only its own shard's pages (block
`B` on shard `B % cp`), so a per-rank partial radix cannot match (rank c has
no block 0). The radix must replicate the token tree across cp ranks while
each block's page stays on exactly one shard, and every rank-local eviction
must degrade to recompute — never cross-shard corruption.

## Hypothesis

A replicated radix (every rank holds the full token tree; `page_id` is
`Some` only on the owning shard) plus a min-reduce exchange of the sealed
extent at publish gives every rank an identical tree with rank-local pages.
The exchange degenerates to a scalar min-reduce because every rank seals the
same rank-identical token stream (SPMD lockstep): rank c contributes
`rank + local_sealed * cp` (the first global block it did not seal), and the
world min is the common prefix every shard sealed. A prefix match walks
through non-owning blocks (emitting a `REPLICA_PAGE` marker); the engine
attaches only the local subset. Residency is per-shard, so the matched
length is min-reduced across ranks — a missing block on any shard truncates
the match for all.

## Parameters

```bash
# A/B: baseline = parent of this commit (prefix cache disabled under 2D),
# treatment = this commit (replicated radix, prefix cache on under 2D).
# Correctness gate (world=4, attn_tp=2, cp=2):
python3 scripts/needle_gate.py --url <url> --model <model> --runs 3
# Prefix-reuse effect (multi-turn / resend prompt share):
#   bench with a resend-heavy prompt set, compare TTFT/prefill tokens.
```

- Baseline: parent of the Part C commit (Part B pool sharding, prefix cache
  disabled under 2D)
- Treatment: Part C commit (replicated radix + attach filter)
- Trials: 3 (needle ladder ×3, same config)

## Environment

- Host / GPU: 8×H20 pod (sm_90), world=4 (attn_tp=2, cp=2)
- Driver / CUDA: TBD
- Model / dtype: Qwen3.5/3.6 hybrid, BF16 (the 2D candidate family)
- TP / EP / slots / KV: attn_tp=2, cp=2, prefix cache on
- Server flags: 2D engaged (world ≥ 4, attn_tp ≥ 2, cp ≥ 2)

## Results

| arm | needle ladder | errors | garble | prefix hit rate | delta |
|---|---|---:|---:|---:|---|
| baseline | | | | 0 (disabled) | — |
| treatment | | | | | |

Raw artifacts: TBD.

## Problems

None yet. Known v1 limitations (by design):

- **Stale replicas under eviction.** A block evicted on its owning shard
  stays as a replica node on peers until the next publish exchange. A peer's
  match walks the stale replica; the cross-rank min-reduce of the matched
  length truncates all ranks to the owner's shorter match. Occasional
  recompute, no corruption. A residency bit vector in the publish exchange
  is the follow-up.
- **Hybrid sidecar restore under 2D degrades to recompute.** The backend's
  `restore_recurrent_sidecar` mirrors `prefix_pages[..need]` with `need` a
  global block count; the host now passes the local subset, so the backend's
  coverage check fails and the attach falls back to full recompute. Backend
  2D-awareness (Part D/E or follow-up) restores decode reuse.
- **Admission budgeting is aligned through the clamp.** The admission peek
  (`try_admit_front_waiter`) matches before attach; its matched length is
  min-reduced inside `clamp_prefix_to_backend` (gated to `kv_shard_factor >
  1`), so the admit/throttle decision sees aligned lengths. Without it a
  rank-local eviction diverges the page budget and desyncs the SPMD
  admission loop.

## Learnings

pending-remote. Design points that held up:

- **Location is a pure function, not a table.** Since `block_size ==
  page_size`, block `B` always lives on shard `B % cp` — zero-storage
  location, and the radix needs no per-block shard column. The same
  predicate single-sources the pool's alloc filter (Part B), the publish
  seal, the attach filter, and the match's local/replica split.
- **The exchange is a min-reduce, not an all-gather.** Every rank seals the
  same rank-identical token stream, so exchanging tokens would exchange
  identical arrays. The only rank-local information is the sealed count,
  and the common sealed prefix is `min_c (rank_c + k_c * cp)`. One scalar
  `tp_sync_min` per publish batch (lockstep at request completion).
- **Replica nodes must not pin eviction.** A resident block whose only
  descendants are replicas must evict; otherwise every other block in a
  chain (cp=2) freezes its ancestor. A per-node `resident_below` counter
  (resident descendants, maintained on every `page_id` transition) makes
  the evictable check O(1) and exact — the old one-level "children all
  demoted" check cannot see through a replica to a live local descendant
  deeper down.
- **The attach filter is the corruption boundary.** `local_block_ids`
  before every pool/backend touch (`retain_pages`, `attach_pages`,
  `restore_prefix_sidecar`) is what keeps a foreign page id — meaningless
  on this rank, able to alias a real local page — out of the slot table.
  It lands regardless of replication completeness: with prefix cache
  disabled under 2D, a stale radix hit degrades to recompute, not
  corruption.
