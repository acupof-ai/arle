# Write-through tiered KV memory (the real infinite-memory)

Supersedes the swap (mid-decode demote/promote) model in
[`2026-06-23-session-infinite-kv-memory.md`](2026-06-23-session-infinite-kv-memory.md).
That model is **killed**: mid-decode tier I/O contends with the single-allocator
`KvPool` — the exact wall both backend implementations hit. The recall *mechanism*
(per-step restricted attention) shipped as a compute-only, default-off variant; the
**memory win** lands here instead.

Decided 2026-06-23 (ckl): **don't swap. Write-through cache, evict-only HBM, prefetch
only at prefill, no tier interaction during decode.**

## Principle — four timing rules

HBM is a **write-through cache** of the session's KV; the tier (host DRAM → NVMe) is
the **source of truth** holding everything. The only rules:

| when | action | why it's free |
|---|---|---|
| **write** (prefill/decode produces a KV page) | mirror HBM→tier **async**, off the critical path | tier always has a copy |
| **HBM full** | **drop** the coldest unpinned page (淘汰) — no write-back | already write-through'd |
| **prefill** | **prefetch** relevant history tier→HBM (the recall) | the one known sync point |
| **decode** | append + attend resident; evict-drop if over budget; **no synchronous tier read** | no latency, no allocator contention |

Why this beats swap: eviction costs nothing (no write-back), prefetch happens at a
single batched point (prefill, not per-step), and decode never blocks on tier I/O or
fights the page allocator. This is the HiCache pattern (write-through L1 + prefetch),
and it dissolves the blocker rather than working around it.

## What the model gives

- **Memory win**: HBM holds a bounded working set (sink + local + prefetched-relevant);
  a million-token session lives in the tier. Flat VRAM vs history — the thing to
  measure on the pod.
- **Recall**: at each turn's prefill, the relevant past is *prefetched* into HBM
  (scored by mean-key relevance to the prompt), then decode attends the resident set.
  Recall moves from per-decode-step → **per-prefill/per-turn**, which fits the
  turn-based agent workload exactly.
- **Boundary (accepted)**: a single ultra-long *generation* with no re-prefill can only
  attend what was prefetched + the growing local window (no mid-decode prefetch). Agent
  turns re-prefetch each turn, so this is a non-issue for the target workload.

## Architecture — where each piece lives

```
infer-seam      KvTier contract: write_through(page) · evict(page)=drop · prefetch(block)->page
                  (replaces demote_block/promote_block-as-swap; same kv-native-sys backing)
infer-core      session block table (rep, tier-loc, last-access) · prefill prefetch policy
                  (reuse recall.rs plan_recall/reps to score) · LRU+pin eviction policy
infer-cuda      paged pool: page = write-through/evict/prefetch granularity (NATURAL)
infer-metal     flat session KV → must become paged/windowed to evict at page grain
kv-native-sys   the L2 (host DRAM) / L3 (NVMe) store, session-keyed (multi-tenant isolation)
```

The existing **RadixCache + host-RAM tier + disk spill** ("KV stays hot across turns")
is the substrate — this *extends* it: write-through (copy proactively, not only under
pressure) + **relevance-prefetch** (not only exact-prefix-hit promote). Its
`demote_block`(→ becomes evict-drop, since write-through already copied) and
`promote_block`(→ becomes prefetch) primitives mostly carry over; the new parts are the
proactive write and the relevance scoring.

## Data structures

Per session (keyed by `session_id` → tenant isolation):

- **Block table**: for each recall-block `b` → `{ rep: [nkv,hd] f32 (resident),
  tier_key, hbm_page: Option<page>, last_access }`. `hbm_page=Some` ⇒ resident;
  `None` ⇒ tier-only (evicted), still scorable via `rep`.
- **Pins**: `sink` (first `n_init`) and `local` (last `n_local`) never evict.
- **Reps pool** (the recall horizon — see refinement R1).

## Flows

**Prefill (turn N):**
```
1. prefix-match (RadixCache, token-keyed) → reuse resident prefix pages.
2. q_rep = representative query of the prompt (R3).
3. for each non-resident historical block b: score[b] = q_rep · rep[b].
4. select via plan_recall(cfg) → top-k blocks; prefetch their KV tier→HBM
   (evict-drop coldest unpinned to make room).
5. pin sink; run prefill; pin the resulting local window.
```
**Decode (tier-free):**
```
per step: append new token KV (write-through async) → attend resident working set.
if filling a page would exceed the HBM budget: evict-drop the LRU unpinned page.
no synchronous tier read.
```

## Refinements (the 完善)

- **R1 — reps don't scale free.** One `[nkv,hd]` f32 rep per block. At page grain
  (16 tok) a 10M-tok session = ~2.5 GB of resident reps (Qwen nkv=8,hd=128) — too much.
  **Decision: recall-block = 256 tokens** (coarser than the page) → ~160 MB / 10M tok,
  acceptable; trade slightly coarser recall for a bounded rep pool. Cap the rep pool;
  when full, drop reps for the coldest blocks → those become prefix-only-recallable
  (graceful horizon, not a cliff). `l_bs` decouples from `page_size`: l_bs=256, page=16.
- **R2 — L2/L3 split.** Write-through to **L2 (host DRAM)** first (fast); a background
  job demotes cold L2→**L3 (NVMe/remote)** when L2 fills. Prefetch reads L2 (fast) or L3
  (slow). Recent history in L2, old in L3 — matches access skew.
- **R3 — prefetch query.** At prefill, score historical blocks with the **mean of the
  last `m` prompt tokens' queries** (the "what am I about to generate" signal), not the
  whole prompt (dilutes). Start `m=16`; tune. Stale-Q is already licensed, so a
  one-prefill-old signal is fine.
- **R4 — async write-through.** CUDA: D2H copy on a side stream into a pinned host ring,
  drained to L2/L3 by a writer thread; never on the decode stream. Metal: unified
  memory makes the "copy" a cheap buffer handoff. Write-through must never stall decode;
  if the queue backs up, apply backpressure at prefill (a known sync point), not decode.
- **R5 — reconcile with RadixCache.** Prefix pages are *already* a write-through-ish
  tier (demote under pressure, promote on prefix hit). Unify: the prefix tier and the
  recall tier are one session-keyed store; prefix-hit promote and relevance-prefetch are
  two entry points to the same `prefetch`. Don't build a parallel store.
- **R6 — rep compute.** Compute a block's rep (mean-pool its K over l_bs) **at
  write-through time** while K is resident — never re-read from tier to score. The
  per-step C++ query emit (already shipped) is reused only at prefill now.
- **R7 — correctness under eviction.** RoPE is pre-baked in cached K, attention is
  position-absolute (verified for the per-step path) — a prefetched block attends at its
  original positions. Same property holds; the prefetch just changes *which* pages are
  resident, not the math.

## Backend specifics

- **CUDA (do first — natural + pod-testable).** Paged pool: page = write-through/evict/
  prefetch unit. write-through = async D2H of a filled page; evict = free the device page
  (the page table already addresses by id); prefetch = alloc page + H2D from tier + write
  its id into the prefill page table. Reuses `decode_graph` page-table machinery and the
  existing `demote_slot`/`promote_prefix_pages`/`kv_tier_*`. Pod-test flat-VRAM on the
  last 4×H20 (Qwen3.5-dense bf16).
- **Metal (second).** Flat session `kv_flat` can't evict at page grain → restructure to
  a **paged or ring-windowed** session KV so cold pages free. Mac unified memory mutes the
  HBM win, so this is about *correctness parity* + supporting longer-than-budget sessions,
  not raw VRAM savings. Validate the working-set bound + needle locally.
- **DSv4 / Qwen3.6-hybrid** own per-slot KV (no paged pool). Bringing them into recall =
  giving them a paged pool, or a separate MLA-aware path (DSA is already sparse — recall
  may be redundant). **Deferred + flagged**; decide per-model after CUDA-dense lands.

## Phasing / DAG

1. **Seam `KvTier` contract** (write_through / evict-drop / prefetch) — replaces the swap
   API. Device-neutral. ← start.
2. **CUDA**: write-through + evict + prefill-prefetch on dense Qwen3 (paged). Reuse the
   page table + tier. **Pod-test flat-VRAM-vs-history** (4×H20) — the decisive evidence.
3. **infer-core prefetch policy** (reps@write-through + prompt scoring + LRU/pin) — shared.
4. **Metal**: flat→paged/windowed session KV + the same contract; local needle.
5. Reps-pool cap (R1) + L2→L3 demote (R2) hardening.
6. DSv4 / Qwen3.6 coverage decision.

## Gates

- **Correctness**: correct-inference needle on a session **exceeding the HBM budget** —
  the prefetch must retrieve. NOT byte-identity (eviction deviates from a single-resident
  run). x3 same-config vs the baseline envelope.
- **Memory (the win)**: `nvidia-smi` flat-VRAM vs growing session length on the pod —
  HBM stays bounded while the session grows to millions of tokens. This is the §6
  decisive evidence and what the swap model could never show.
- **Baseline**: default-off → byte-identical (CUDA is Stable). The opt-in is the same
  `--kv-recall` flag, now meaning write-through-tier + prefetch.
- **Multi-tenant**: session A's tier never prefetches into session B.
