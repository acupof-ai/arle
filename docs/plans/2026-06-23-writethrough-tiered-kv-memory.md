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

| when | action |
|---|---|
| **write** (page completes) | keep the block's mean-key rep resident; KV stays in HBM — **L2 is write-back**, populated on eviction (NOT continuously mirrored → cheap hot path) |
| **HBM full / evict** | **write-back** the cold block HBM→**L2** (host DRAM, fast); free the HBM page (event-ordered — never while a kernel still reads it) |
| **L2 → L3** (async, off the decode path) | **write-through** to **L3** (NVMe): every L2 block is durably in L3 → no block is ever lost |
| **prefill** | **prefetch** relevant history L2/L3→HBM (the recall): L2 hit = fast, L3 = slow |
| **decode** | append + attend resident; write-back-evict if over budget; **no synchronous tier read** |

**Per-tier write policy (ckl 2026-06-23): HBM→L2 write-back** (lazy, on eviction — the hot
path never continuously mirrors) **+ L2→L3 write-through** (every evicted block lands
durably in L3). `L3 write-through` = **full recall coverage** — a block is never
dropped/lost (unlike the resident-only arm that drops on evict), so recall can always
re-fetch. `L2 write-back` = cheap eviction into fast DRAM. Decode never blocks on tier
I/O; prefetch is the one batched sync point (prefill). This beats swap: no mid-decode
tier read, no per-step allocator fight.

## Global budget ledger + swap/staging area (ckl 2026-06-23)

One HBM ledger, **fully pre-reserved** — no on-the-fly alloc (on-the-fly was the
34.4 GB / 524288-token over-alloc seen on the pod). **Timely (proactive)** eviction
holds the invariant.

```
HBM KV budget (resource.rs kv_capacity) =
    working_set   sink + local + recalled      (bounded to working_set_tokens)
  + staging       >= 1 KV-block upper-bound     (prefetch-in landing + write-back-out source)
  + reps          resident mean-key reps        (rep-pool capped)
```

- **HBM 交换区 (staging)**: reserve >= one KV-block so a prefetch (L2/L3->HBM) lands and
  a write-back (HBM->L2) stages **without first evicting** and **without a surprise
  malloc** (malloc implicit-syncs + can OOM). Decouples prefetch from evict timing.
- **DRAM (L2) staging**: same — reserved landing for HBM write-backs + source buffer for
  the L2->L3 write-through.
- **Timely eviction**: proactively write-back-evict to keep
  `working_set <= budget - staging - reps` — always >= 1 free block for append/prefetch.
  Never lazy-too-late (that blocks or OOMs).
- **Why global**: eviction timing × staging reuse × transfer event-order (write-back
  done before staging is reused; prefetch landed before attend) must be **one ledger +
  one policy**, not per-call. This is the hard part.

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

## Where it must land: the rollout engine (train agent-opd) — 4 gaps (ckl 2026-06-23, grounded)

The real "infinite memory" target is the **OPD agent rollout**, not just serve
`--kv-recall`. The rollout engine today (grounded read):

1. **Cap = hard, build-time, HBM-preallocated**: `total_pages = num_slots ×
   (max_turns×max_tokens + 8192) / 16`, claimed once at engine build. max_seq_len
   too big → **build OOM** (the one I hit). No dynamic growth.
2. **Eviction = decode preemption only** (`retract_decode_to_fit`, per scheduler
   tick): pool full → retract the shortest-generation decode row to the queue =
   its generated tokens **dropped + recomputed** (token loss). Prefill never preempted.
3. **No tier/swap wired for rollout**: T1(host-RAM)/T2(SSD) infra exists but only in
   serve mode; the train agent-opd engine build doesn't wire it — not even
   `--kv-ssd-path`. Pool full → no overflow path, only preempt-recompute.
4. **RadixCache LRU is decoupled from engine pages**: evicts prefix-cache blocks,
   doesn't free engine pages — not a pool-pressure lever.

| want | rollout now | how the tier closes it |
|---|---|---|
| bounded cap | ✅ but dead/HBM-prealloc → OOM when grown | HBM = bounded working set (sink+local+recalled+staging); build sizes to that, not max_turns×max_tokens → no build OOM |
| timely eviction | ⚠️ reactive preempt = drop+recompute | evict = graceful **write-back-demote** to L2/L3 (keepalive + write-through, verified on serve) → zero token loss |
| reserved swap | ❌ rollout has none | **wire the tier into the rollout engine build** + the reserved staging |
| admission / per-req cap | ❌ none — hits the wall then preempts | add **admission control / per-request KV cap** at the scheduler (block at entry) |

**Order**: ✅ mechanism (serve recall + L3 tier — GPU-verifying on GPU 0) → wire tier
into the rollout build → graceful-demote replaces preempt-recompute → bounded-working-set
cap (kills build OOM) → admission control. The mechanism is the same code; the work is
**wiring it into train agent-opd's engine build + the scheduler**, which serve already has.

## Systematic design: budget · eviction · staging · prefetch (grounded 2026-06-23)

The four axes are **one interlocked system**, not four features:

> a **global per-tier budget ledger** fixes HBM headroom → **proactive timely eviction**
> cascades L1→L2→L3 to hold that headroom → a **reserved staging area** makes every
> transfer bounded + async (no surprise malloc, no evict-before-prefetch ordering) →
> **prefetch at the prefill sync point** refills the working set by rep-relevance.

Each below: **target** · *grounded current state (file:line)* · **the change**.

### ① Budget — one ledger, per tier, sized to the working set (NOT max_seq_len)
*Now: `recall_kv` token budget = `num_slots × max_seq_len` (`executor.rs:3065`) = the
34.4 GB over-alloc; no ledger; rollout sets `total_pages = num_slots × max_seq_len/16`
(`train_cli.rs:1547`) with no free-VRAM clamp → build OOM. Tiers (HBM/T1/T2) are
independent capacities, not a ledger.*

```
HBM_ledger = working_set + staging + reps        (all bounded; none scales with max_seq_len)
  working_set = num_slots × (n_init + n_local + top_k·l_bs + headroom) pages   ← the bound (~544 tok/slot)
  staging     = num_slots × ceil(1 KV-block)                                    ← ②
  reps        = num_slots × max_blocks × rep_bytes   (f32[kv_heads·head_dim]/l_bs-block; rep-pool capped)
L2(DRAM) budget = the session-history cap (the real "256K→millions")            (default_t1, kv_tier.rs:71)
L3(NVMe) budget = durable overflow                                              (default_t2)
```
- Change `executor.rs:3065` token budget `num_slots×max_seq_len` → `working_set_tokens`.
- Change `train_cli.rs:1547` to the same bounded sizing + the `kv_budget_num_slots`
  clamp (`qwen35.rs:1636`) → **kills the build OOM**.
- `cuda_admission_total_pages` (`loaded.rs:1266`) reserves working_set+staging up front
  (recall pool no longer a lazy after-the-fact alloc that competes with slots).

### ② Eviction — proactive + graceful, cascading L1→L2→L3
*Now: preemption (`planner.rs:94-182`) IS graceful-demote when a tier is wired
(publish→demote→free), else recompute; recall evict-drop (`decode_row_recall`,
`executor.rs:3319`) write-throughs then keepalive-frees, coldest-unpinned
(`writethrough.rs:114-148`). Both **reactive (at-the-wall)**; rollout has no tier → recompute.*
- **Proactive**: evict to hold `working_set ≤ budget − staging − reps` at a low-watermark
  each tick (BEFORE the wall) → append/prefetch always has room. New headroom check in
  the scheduler tick / `decode_row_recall`, not only `retract_decode_to_fit`'s at-wall trigger.
- **Cascade**: L1→L2 **write-back** on evict (cold block → DRAM); L2→L3 **write-through**
  (every L2 block durably to NVMe, async). Reuse the coldest-unpinned + sink/local-pin
  policy (`writethrough.rs:114`), extend to the cascade.
- **Graceful in rollout**: wire the tier so `requeue_preempted_decode` takes the demote
  path, not the recompute fallback → zero token loss.

### ③ Staging — reserved swap area, pinned, async (decouples transfer from evict)
*Now: NO reserved staging. D2H allocs a `Vec` on-the-fly (`copy_pages_to_host`,
`paged_kv.rs:869`); H2D borrows; copies are synchronous; no pinned-host ring (R4 pending).*
- Reserve in the ledger: **HBM staging** ≥1 block/slot (prefetch-in landing + write-back-out
  source-pin); **DRAM staging** = a pinned-host ring (`cudaHostAlloc`) — the D2H landing +
  H2D source.
- D2H/H2D (`paged_kv.rs:869/953`) read/write the ring on a **side-stream**, not the decode
  stream (the R4 async mirror) → no per-copy `Vec`, no decode stall.
- **Decouples** prefetch from evict: prefetch lands in the reserved staging page (no
  "evict first to make room") → no ordering deadlock. Keepalive handles the evict race;
  staging handles the prefetch landing. Both fixed-size → no surprise malloc/OOM.

### ④ Prefetch — at the prefill sync point, batched, reps-scored
*Now: per-**decode-step** (`executor.rs:3291`, mid-decode H2D), contradicting the plan;
violates "no mid-decode tier read"; rollout doesn't wire the tier so the path is dead there.*
- Move to the **prefill / per-turn** sync point: at each turn's prefill, score the whole
  history (reps · prefill query, `recall.rs:280`) → top-k → **one batched H2D** into the
  working set via the staging ring; decode then attends the resident set (no mid-decode read).
- Fits the rollout (multi-turn agent: each turn re-prefills = the natural batched point).
  Keep the per-step arm only for a single ultra-long generation with no re-prefill.

### Implementation order (each lands + verifies independently)
1. **Budget ledger** (`executor.rs:3065`, `train_cli.rs:1547`, `cuda_admission_total_pages`)
   — kills the 34.4 GB over-alloc + the build OOM. Foundation.
2. **Reserved staging** (HBM page + pinned-host ring; `paged_kv.rs:869/953`) — bounded async transfer.
3. **Wire tier into rollout** (`train_cli.rs` build + `set_disk`) — makes preemption graceful (gap ③).
4. **Proactive eviction** (headroom watermark; scheduler tick / `writethrough.rs`) — gap ② fully.
5. **Per-prefill prefetch** (move off `decode_row_recall:3291` to the prefill path) — the plan intent.
6. **Admission control** (scheduler entry-gate / per-request KV cap) — gap ④.
