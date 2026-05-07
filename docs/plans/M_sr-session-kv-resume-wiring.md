# M_sr — Session KV checkpoint + resume wiring

> Discovered 2026-05-07 from dead-code audit. ARLE has full
> substrate for "save session KV → resume session with intact
> KV cache" (4 public APIs in `scheduler/cuda/core.rs:217-269`)
> but zero production callers. Second dead-code-substrate gap
> found this session (first: `submit_prefetch_plan` → M_pf).

## Priority & ROI

**Priority**: **P1 license-or-kill** (Phase 1A v3 fix has now
shipped — codex `5cacdcb` default Split. M_sr is no longer
gated on it). Higher priority than M_pf P3 because the gain
is much larger for ARLE's "world-first long-sequence" project
goal: deterministic 58× TTFT savings on agentic resume vs M_pf's
5-15% TTFT.

**ROI basis**:

For agentic / chat resume scenarios (typical of long-sequence
production):
- 32k-token accumulated session KV: 32768 × 36 layers × 2 (K+V) ×
  8 KV heads × 128 head_dim × 1 byte (FP8) = **2.4 GB host KV**
- Save on idle (D2H): ~120 ms at PCIe 4.0 (~32 GB/s, contiguous)
- Resume on next message (D2H from host pool back to GPU): ~120 ms
- Without session resume: full re-prefill of 32k tokens at
  0.21 ms/token = ~7 seconds
- **TTFT savings on resume: 7s → 0.12s = 58× faster** (or "the
  user's question gets answered ~7 seconds faster after coming
  back from idle")

For high-conc / single-turn workloads: zero benefit.

**Substrate-cost**: implementation = wire-up only. ~150-250 LOC:
- HTTP `session_id` already in `IncomingRequest` (cooperative path
  exists)
- Disk persistence: `DiskStore` already exists
- Read/install primitives: `read_block_payload` / `install_restored_kv`
  exist
- Missing: scheduler intake hook to load on session-id match,
  scheduler eviction hook to save on session-idle

**Why this is feasible despite complexity**:
- All HARD parts already done by prior work
- Wiring is "compose existing primitives" not "design new ones"
- Tests for session_id correctness already exist (M_d.1
  isolation tests via fingerprint namespace)

**Negative case**:
- No-session workloads (single-turn LLM API): zero benefit
- Disk pressure if many sessions persist simultaneously
- Resume latency could be worse than re-prefill if disk is slow
  (e.g. NFS / spinning rust)
- Risk of stale KV: tokenizer/model swap mid-session would
  serve old KV (M_d.1 namespace defense partially covers)

**Kill criteria**:
- Phase 0 license bench: agentic resume scenario shows < 10×
  TTFT improvement (vs naive re-prefill) → ABANDON
- Disk write rate during normal load > 20% of compute time
  (overhead too high) → revert
- Any e2e regression on non-session workloads → revert
- LOC overrun: implementation > 400 LOC → revisit
  (might mean substrate is incomplete)

## Discovery

`grep -rln` for callers of public APIs in
`infer/src/scheduler/cuda/core.rs` revealed:

| API | Defined at | External callers |
|---|---|---:|
| `read_block_payload` | core.rs:217 | **0** |
| `install_restored_kv` | core.rs:225 | **0** |
| `session_disk_store` | core.rs:264 | **0** |
| `session_radix_cache` | core.rs:268 | **0** |
| `session_fingerprints` | core.rs:300 | **0** |

These 5 functions form a complete checkpoint/resume API:
- `read_block_payload(fingerprint) → Option<Vec<u8>>` — D2H by
  fingerprint
- `install_restored_kv(payloads: HashMap<Fingerprint, Vec<u8>>) →
  Box<dyn FnMut(Fingerprint) → Option<BlockId>>` — bulk H2D restore
- `session_disk_store() → &DiskStore` — persistence layer accessor
- `session_radix_cache() → &RadixCache` — fingerprint accessor
- `session_fingerprints(session_id) → Vec<BlockFingerprint>` —
  list a session's blocks

Source comments + structure suggest this was built during the
"Tiered KV Cache" project but the integration tick never landed.

## Phase 0 — license-or-kill experiment (~30 min, ~1-2 hr GPU)

### Construct minimal session-resume bench

1. Boot ARLE with session_id support
2. Send long prompt (32k tokens) under `session_id="alice"` →
   measure baseline TTFT
3. Send follow-up message under same `session_id` after 60s idle
4. Today (no resume wired): TTFT will include full prompt re-prefill
   if cache evicted, OR partial cache HIT if not.

To create the worst-case (no cache HIT):
- Force eviction by submitting many other requests in between
- OR set `prefix_cache_high_water=0.5` to encourage eviction

If even today's RadixCache covers the resume case (no eviction
under typical pressure) → **abandon M_sr** (existing prefix_cache
already adequate).

### License decision

- PROCEED if observed TTFT for "post-idle resume" > 10× the
  expected ~120ms D2H restore time
- ABANDON if RadixCache + paged_kv_pool already covers the
  scenario (i.e. resume TTFT < 1 second naturally)

## Phase 1 design (only if licensed)

### P1.1 — Idle-session save hook (~80 LOC)
At session timeout / explicit logout, fire D2H copy of all
session blocks to DiskStore. Async on copy_stream.

### P1.2 — Resume-on-incoming hook (~80 LOC)
At HTTP request entry with session_id, query DiskStore for
matching session. If found, fire `install_restored_kv` on the
fingerprints. Block until restore complete OR fall back to
re-prefill.

### P1.3 — Eviction policy update (~30 LOC)
RadixCache eviction must respect "session_id has disk-backed
copy" — don't evict before disk persist completes.

### P1.4 — Bench (~0 LOC)
Multi-turn agentic-style session: 32k context, 8 turns spaced
60s apart with idle injection. Compare: no M_sr (re-prefill)
vs M_sr (D2H restore). Expect 30s+ latency savings per turn.

## Tasks

| # | Task | File | LOC | Owner | Trigger |
|---|---|---|---|---|---|
| Phase 0.1 | Construct session-resume bench script | `scripts/bench_session_resume.py` (new) | ~80 | Claude | Now (Phase 1A v3 already shipped) |
| Phase 0.2 | Bench scenario, measure resume TTFT | bench | 0 | Claude | Phase 0.1 done |
| Phase 0.3 | License decision (PROCEED if > 10× expected D2H, else ABANDON) | analysis | 0 | Claude | 0.2 done |
| P1.1 | Idle-session save hook | `infer/src/scheduler/cuda/runtime/admission.rs` (or session_slots.rs) | ~80 | Codex | License fires |
| P1.2 | Resume-on-incoming hook | HTTP entry path | ~80 | Codex | P1.1 |
| P1.3 | Eviction respects disk-persist | `infer/src/prefix_cache.rs` | ~30 | Codex | P1.2 |
| P1.4 | Multi-turn session bench validation | bench | 0 | Claude | P1.1-3 commit |

## Cross-references

- Discovery: dead-code audit 2026-05-07
- Source: `infer/src/scheduler/cuda/core.rs:217-269`
- M_pf (first dead-code-substrate) precedent:
  [`544d00d`](M_pf-radix-prefetch-wiring.md)
- M_d.1 namespace (correctness for cross-session): `0e1bc3d`
- Tiered KV Cache project: `docs/projects/tiered-kv-cache.md`

## Rule (per memory `feedback_docs_priority_roi_evidence.md`)

- Includes Priority + ROI + Negative + Kill criteria
- Phase 0 is the license-or-kill experiment
- Per-phase ROI assessed (Phase 0 = decision quality;
  Phase 1 = TTFT improvement on session-resume workloads)
- This is the **second** dead-code-substrate finding this
  session (first: `submit_prefetch_plan` → M_pf with 1:30 ROI
  on the experiment vs implementation savings)
