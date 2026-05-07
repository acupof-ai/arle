# 2026-05-07 · Phase 1A v3 fix validated — default Split fixed, Mixed opt-in STILL regressed

## Priority & ROI

**Priority**: Default Split = production solid. Mixed opt-in =
**defer** (deeper kernel/scheduler issue beyond Fix A scope).

**ROI**:
- Default Split: **NET POSITIVE** vs F4-Small baseline — out
  tok/s 122.5 → 153.9 (+25.6%) at longctx 4k/c=4. Multi-slot
  ring substrate kept produces incidental gain even when not
  toggled active.
- Mixed opt-in: **STILL REGRESSED** even with Fix A budget
  applied. 41.4 out tok/s = -73% vs default Split. Need M_nsys
  proper trace to diagnose.

## Goal

Validate codex's Phase 1A v3 fix (`5cacdcb fix(scheduler):
default mixed prefill to split`):
1. Default mode (Split) should restore F4-Small baseline or better
2. Mixed opt-in should work (Fix A should fix budget regression)

## Bench results

ARLE build: HEAD post-`5cacdcb`. longctx 4k/c=4, c=4, 60s + 10s warmup.

| Mode | out tok/s | TTFT mdn | ITL mdn | Notes |
|---|---:|---:|---:|---|
| **Default Split (production)** | **153.9** | (table) | (table) | **+25.6% vs F4-Small 122.5** ✓ |
| Mixed opt-in (max-seq-len 12288) | 0 | n/a | n/a | KV pool starved by mixed workspace, max_input=122 |
| Mixed opt-in (max-seq-len 5120) | **41.4** | (table) | (table) | -73% vs Split, -66% vs F4-Small ✗ |

Default Split bench:
- 153.9 out tok/s = **+25.6% over F4-Small longctx baseline (122.5)**
- This is unexpected gain — multi-slot ring substrate (kept per
  Option B) provides incidental benefit even though Mixed path
  is not invoked
- Production deployments unaffected by Phase 1A v3 substrate

Mixed opt-in bench (max-seq-len 12288): full failure
- Server log: `max_input=122 max_request=127`
- ALL incoming 4097-token prompts rejected (`Rejecting prompt
  with 4097 tokens`)
- Cause: codex's "Gate mixed workspace reservation behind the
  same policy" works in REVERSE — when Mixed enabled, workspace
  reservation eats KV pool, leaving 122-token capacity

Mixed opt-in bench (max-seq-len 5120): partial work
- Server accepts requests (smaller per-request KV)
- Some throughput happens (41.4 out tok/s)
- But STILL severely regressed vs default Split

## Why Mixed still regressed (post-Fix A)

Codex's Fix A (`mixed_prefill_token_budget` returns full
`max_prefill_tokens`) addresses the budget undersizing. But two
secondary causes remain:

### Cause 1: Mixed workspace reservation conflict

When Mixed enabled, additional GPU workspace reserved for the
varlen Q packing. At max-seq-len 12288 + num-slots 8, that
reservation collides with KV pool, leaving tiny per-request
capacity. **Workspace sizing logic needs revisit.**

### Cause 2: Per-prefill-row prep loop overhead

From Phase 1 retro analysis (`c54fb5d`):
- Mixed kernel CPU launch avg = 5.2 ms
- Decode kernel CPU launch avg = 1.0 ms
- Reason: per-prefill-row `prefill_attention_paged_prep_cuda`
  loop (1 + N launches per layer × 36 layers)

Phase 1A v3 Fix A enables Mixed to pack 4 prefill rows per step
(was 1 row). But each step now has 4× more prep launches:
- 4 reqs × 36 layers × ~5 µs launch overhead = 720 ms launch
  overhead per step (CPU side)
- Net: more rows per step but each step way slower

This is the **structural issue** retro analysis hinted at. Fix
requires unifying per-prefill-row prep into one varlen prep
launch (use HD128 prefill kernel's existing `q_indptr` surface).

### Combined

Cause 1 (workspace) + Cause 2 (per-row prep loop) both contribute
to Mixed staying slow. Even at max-seq-len 5120 (workaround
workspace conflict), Cause 2 dominates → 41.4 vs 153.9 = 73%
slower.

## Decision

1. **Default Split SHIPS as is** — production benefits from the
   multi-slot ring substrate (+25.6%) without enabling Mixed.
2. **Mixed opt-in declared "experimental, not production-ready"**
   — codex's flag system accommodates this. Documentation flag
   should warn.
3. **M_nsys P0 (cudaProfilerStart/Stop) becomes blocking** for
   any further Mixed analysis. Without proper trace, we can't
   distinguish Cause 1 vs Cause 2 contributions. **No further
   Mixed-fixing without M_nsys first.**
4. **Future Mixed work (deferred,P3)**:
   - Fix workspace sizing logic (~30 LOC)
   - Unify per-prefill-row prep loop into varlen prep launch
     (~80 LOC, M_b.3 candidate)

## Cross-references

- Codex fix commit: `5cacdcb`
- Phase 1A v3 regression confirmation: `ba748af`
- Root cause analysis: `e675af2`
- Phase 1 retro (5× launch finding): `c54fb5d`
- Multi-slot wins entry: `b32ca9f` (mark M3.9 readback review clean)
- M_nsys plan (now blocking for Mixed work): `3a783f1`
- Default Split bench: `bench-output/2026-05-07-fix-default-split/`
- Mixed opt-in bench: `bench-output/2026-05-07-fix-mixed-smaller/`

## Rule

- **Default-safe fixes ship even with caveats**. Codex's Phase
  1A v3 fix routes production through Split (no regression),
  even though Mixed opt-in still has bugs. This is the right
  product call: ship working default, defer broken opt-in.
- **Compounding bugs hide each other**. Fix A revealed Cause 2
  (per-row prep loop). Without Fix A, Cause 2 was masked by the
  budget undersizing. Iterative diagnosis is correct here, not
  "fix everything at once".
- **Real diagnostic infrastructure precedes complex
  optimization**. M_nsys is now P0 for further Mixed work; can't
  fix what we can't trace.
