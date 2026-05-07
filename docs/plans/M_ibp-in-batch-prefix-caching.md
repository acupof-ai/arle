# M_ibp — In-batch prefix caching (simultaneous-request prefill dedup)

> Identified 2026-05-07 from user question "In-batch Prefix Caching
> has done?". ARLE has cross-request RadixCache (sequential cascade
> via sealed-block publish) but **NO** simultaneous-request prefill
> dedup within the same batch. M_ibp closes this gap for the
> multi-tenant RAG / agentic workload class.

## Priority & ROI

**Priority**: **P2** (after P0 M3.9 Phase 1A v3 + P1 M_pf land).

**Why P2 not P0/P1**:
- Distinct from P0 (decode-axis sync) and P1 (cached-prefix prefetch)
- Workload-dependent: only fires when ≥2 simultaneous requests
  share prefix tokens (multi-tenant RAG, agentic batch invocations)
- Single-tenant / heterogeneous-prompt workloads: zero benefit
- Higher implementation complexity than M_pf — touches admission
  scanning + cross-slot KV publishing + radix tree mutation in
  middle of prefill (vs M_pf which only fires existing infra)

**ROI basis** (deterministic for HIT-able workloads):

Worked example: 4 incoming requests, each with 6k shared system
prompt + ~1k user query, c=4 admission burst.

Without M_ibp (today's ARLE behavior):
1. Request 0 admits, prefills chunk 1 (~430 ms), seals + publishes
   blocks 0-127
2. Requests 1-3 wait in queue ~430 ms (admission can't yet HIT
   because chunk 1 hasn't sealed yet)
3. Once 0's chunk 1 done, admission for 1-3 HITs partial cache
4. 1-3 each prefill remaining (after their chunk 1 reuses 0's): ~430 ms × 2 chunks each = 860 ms more
5. 0 also continues chunks 2-3: ~860 ms total
6. **Total time ~430 + 860 = 1290 ms before all 4 reach decode**

With M_ibp (admission-side simultaneous-request prefix dedup):
1. Admission scan: 4 requests share 6144 tokens (3 chunks of 2048)
2. Issue ONE 6144-token prefill on shared prefix
3. Distribute resulting KV pages to all 4 slots via reference
4. Each slot independently prefills its ~1k suffix (~215 ms)
5. **Total time ~430 ms × 3 + 215 = ~1505 ms before all 4 reach
   decode** — wait, ARLE today actually batches the shared
   computation if cache hit fires fast?

**Correction**: ARLE's implicit "sequential cascade" pattern at
chunked prefill boundary IS roughly equivalent to M_ibp for
boundary-aligned shared prefixes. The gain is smaller than the
naive 4× I'd projected.

Honest re-estimate:
- ARLE today: requests 1-3 wait 430 ms for first chunk, then
  cascade. Total work ~1290 ms.
- M_ibp ideal: requests 1-3 don't wait, all 4 admitted in same
  batch with shared prefill. Total work depends on how mixed the
  prefill is.
- **Realistic gain: 20-40% TTFT improvement** at multi-tenant
  shared-prefix workloads, NOT 4× as pure dedup math suggests.
- For unaligned shared prefixes (system prompts shorter than chunk
  boundary OR with per-request prefix variation), gain is smaller.

**Stacked with M_pf and Phase 1A v3**:
- Phase 1A v3 alone: long-ctx 8k TTFT 4961 → ~1500 ms
- + M_pf (cached prefix): ~1500 - 14 = ~1486 ms
- + M_ibp (multi-tenant shared prefix): ~1486 × 0.7 = ~1040 ms
- = **~2.3× faster than vLLM 2367 ms at multi-tenant shared-prefix
  long-ctx**

**Negative case**:
- Single-tenant: zero benefit (no sharing to dedup)
- Heterogeneous prompts: zero benefit
- Long-prefix-aligned-but-no-multi-tenant: zero benefit
- Risk: admission-side cross-request scan adds latency to ALL
  requests (paid tax for opportunistic gain)

**Kill criteria**:
- Phase 1 bench at multi-tenant shared-prefix workload (4 reqs ×
  6k system prompt + 1k user, c=4) shows < 20% TTFT improvement
  → revert (gain too small to justify admission scan cost)
- Single-tenant bench (1k×1k×c=64) shows > 5% TTFT REGRESSION
  → revert (admission scan tax too high)
- Implementation complexity exceeds 300 LOC → revisit (might
  not be worth the substrate change)

## Hypothesis (testable)

ARLE's existing sequential-cascade pattern (chunk-boundary publish
to radix → next admission HITs) ALREADY captures most of the gain.
Need to MEASURE the gap before committing to implementation.

**Phase 0 — measurement before plan body**:

Add a workload-specific bench: 4 reqs × 6k system prompt + 1k user
query, c=4. Compare:
- ARLE today (after Phase 1A v3 lands)
- vLLM same shape

If ARLE TTFT < 1.5× vLLM at this shape → cascade already covers it
→ **abandon M_ibp** (not worth the complexity). If ARLE TTFT > 2×
vLLM → real dedup gap → proceed.

This is a **license-or-kill experiment**, not an implementation
plan. Phase 1 only fires if Phase 0 measurements license it.

## Phase 1 design (only if licensed)

If license fires, plan body:

### P1.1 — Admission-side cross-request scan (~80 LOC)

In `try_admit_waiting`, when N ≥ 2 requests are pending in the
waiting queue with non-trivial overlap:
- Trie-style merge: build a temporary trie of the N prompts
- Identify the common prefix length P
- If P ≥ chunk_size, fire ONE prefill of the common P tokens
- The other N-1 requests' radix lookups will then HIT (post-publish)

### P1.2 — Cross-slot KV reference (~120 LOC)

Today's RadixCache attaches blocks to slots via `set_block_session_id`
(per-block ownership). Cross-slot reference requires:
- Multi-owner blocks (ref_count ≥ 1 across N slots)
- Slot-private vs shared distinction
- Eviction policy update (don't evict if any owner active)

This is an extension to the existing block_manager / pool layer.

### P1.3 — Hot-tail commit before sealing (~50 LOC)

Today: blocks publish only at chunk boundary (sealed full block).
For M_ibp, we want EARLIER publish — after the FIRST writer's
hot tail starts but before chunk seals — so other waiting requests
can HIT mid-chunk. Risk: hot tail is in-flight; reader could see
inconsistent state.

Mitigation: stable token-id sequence (BPE deterministic) + per-chunk
write barrier on first writer.

### P1.4 — Bench validation

After P1.1-P1.3 land, bench the multi-tenant shape from Phase 0.
Expect 20-40% TTFT improvement.

## Tasks

| # | Task | File | LOC | Owner | Trigger |
|---|---|---|---|---|---|
| Phase 0.1 | Define multi-tenant shared-prefix workload (guidellm shape) | `scripts/bench_guidellm.sh` invocation | 0 | Claude | Phase 1A v3 + M_pf land |
| Phase 0.2 | Bench ARLE + vLLM at this shape | bench | 0 | Claude | Phase 0.1 |
| Phase 0.3 | License decision (ARLE TTFT > 2× vLLM → proceed; else abandon) | analysis | 0 | Claude | Phase 0.2 |
| P1.1 | Admission cross-request prefix scan | `infer/src/scheduler/cuda/runtime/admission.rs` | ~80 | Codex | License fires |
| P1.2 | Cross-slot KV reference (block_manager + pool) | `infer/src/prefix_cache.rs` + `infer/src/scheduler/cuda/core.rs` | ~120 | Codex | P1.1 |
| P1.3 | Hot-tail early publish | `infer/src/scheduler/cuda/core.rs` | ~50 | Codex | P1.2 |
| P1.4 | Bench validation | bench | 0 | Claude | P1.1-3 commit |

**Phase 0: ~30 min Claude work, 1-2 hr GPU time. Phase 1: ~250
LOC (codex 1-2 day) IF licensed.**

## Acceptance

### Phase 0 license
- Workload-specific bench data committed
- Decision: PROCEED if ARLE TTFT > 2× vLLM at multi-tenant shape;
  ABANDON if < 1.5×; investigate if 1.5-2× (might be Phase 1A v3
  edge or other factor)

### Phase 1 (if licensed)
- `cargo test --release -p infer --features cuda --test e2e` passes
- `cargo test --release -p infer --features cuda --test
  greedy_consistency` passes (cross-slot KV correctness)
- New test: `in_batch_prefix_dedup_consistency` — assert byte-exact
  output equivalence between (a) sequential-cascade run and (b)
  M_ibp dedup run on same multi-tenant prompt set
- Bench: ≥ 20% TTFT improvement at multi-tenant shared-prefix
  workload vs no-M_ibp baseline

## Risks + retreat

- **R1 — Sequential cascade already captures the gap**: ARLE's
  chunk-boundary publish + admission re-lookup approximates M_ibp
  at chunk granularity. Phase 0 license-or-kill protects against
  this.
- **R2 — Cross-slot KV reference invariant breakage**: changing
  block ownership from single-slot to multi-slot ref_count breaks
  eviction / retraction / fingerprint contracts elsewhere.
  Mitigation: cargo test --features cuda --test e2e + careful
  invariant audit.
- **R3 — Hot-tail early publish race**: another request sees
  partial KV write. Mitigation: skip hot-tail publishing in v1
  (only commit at chunk seal — gives less gain but no race).
- **R4 — Admission scan latency for non-shared requests**: each
  admission pays scan cost. Mitigation: only scan when waiting
  queue depth ≥ 2 AND prompts have non-trivial token-prefix overlap
  detection (cheap hash-based pre-filter).

## Out of scope

- Disk-tier (T2) prefix sharing: M_ibp v1 is GPU/host-tier only
- Per-token (vs chunk-aligned) prefix sharing: requires deeper
  KV layout changes
- Cross-process prefix sharing: requires shared memory infra
- Spec-decode interaction with M_ibp: handled separately if needed

## Cross-references

- Discovery: user 2026-05-07 question "In-batch Prefix Caching
  has done?"
- Existing cross-request cascade:
  `scheduler/cuda/core.rs:1175-1195` (sealed-block publish comment)
- M-final roadmap: `d16effe` + `a06c5c9` (this slots into the
  "what's missing for world-first" Tier 3)
- Sequential cascade pattern: `try_admit_waiting` →
  `lookup_or_stage` (admission.rs:147-227)
- vLLM v1 reference (their continuous batching has implicit
  in-batch dedup):
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/scheduler.py

## Rule

- **License-or-kill experiments** for hypothesis-stage work
  (Phase 0): cheap measurement protects against committing
  development time to gain that already exists or is < threshold.
  Per memory rule (`feedback_docs_priority_roi_evidence.md`),
  research-y work makes Phase A a cheap experiment.
