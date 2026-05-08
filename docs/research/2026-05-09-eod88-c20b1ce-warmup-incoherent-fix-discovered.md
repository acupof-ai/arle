# `c20b1ce` warmup fix incoherent — bimodal root cause still unverified

> Per codex P0.2 hybrid loader work touching `warmup.rs`,a hidden
> regression discovered:`c20b1ce`'s prefill_cap-aware `max_bs` extension
> was **incoherent**(decode warmup indexes slot-local state by slot id,
> can't warm slots that don't exist)。Codex's P0.2 patch reverts
> `max_bs = num_slots.min(256)`。
>
> **Strategic implication**:bimodal regression root cause analysis
> from `db20d34`(H4)was built on a broken assumption。My P0.3
> Phase 0 audit(`1fdd763`)inherited this without catching it。
> Validates `d2c2c17` strategic brief insistence on Phase 1
> evidence decomposition before P1 commitment。

## The discovery

Codex P0.2 hybrid loader implementation touched `infer/src/scheduler/cuda/core/warmup.rs`
(15-line revert):

**Before(`c20b1ce`)**:
```rust
let prefill_cap = self.model.max_concurrent_prefill_requests().unwrap_or(0);
let max_bs = num_slots.max(prefill_cap).min(256);
```

**After(codex P0.2)**:
```rust
// Warm only batch sizes that can map to real scheduler slots. Some
// models expose a larger prefill cap for admission, but decode warmup
// still indexes slot-local state and paged-KV metadata by slot id.
let max_bs = num_slots.min(256);
```

## Why `c20b1ce` was incoherent

The warmup loop body:
```rust
for slot in 0..max_bs {
    if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
        error!("Warmup: pool alloc for slot {} failed: {}", slot, e);
        break 'warmup;
    }
}
```

If `num_slots=4` and `prefill_cap=8`,`max_bs=8`。Loop tries `slot=4..7`
which **don't exist** as scheduler slots → `alloc_tokens` fails → silent
`break 'warmup` exits early。

→ Warmup actually only succeeds for `slot in 0..num_slots`,then errors
out。Batches > num_slots are NEVER warmed in practice。

## Implications

### Strategic(world-#1 mission)

`c20b1ce` was paired with `db20d34` H4 root cause:
> "Server startup log: 'Warming up CUDA Graphs for 4 batch sizes (max 4)'
> At cap=8 burst, batch=5-8 prefill 触发 first-encounter graph capture"

The "fix" was supposed to extend warmup to batch=5-8。**But it never
actually warmed batch=5-8** — the loop silent-failed at slot=4。

**Yet bench results(reported in `12300c5+c20b1ce`)showed cap=8 stable**。
→ Either:
- (a) The cap=8 stability came from something OTHER than warmup
  (cublasLt heuristic from previous bench runs persisting,or eager
  warmup fallback,or actual workload not triggering batch=5-8)
- (b) Bench measurement was post-warm-up via warmup itself(despite
  failure)— first-burst behavior in fresh server still cold

→ **The 33% degraded path bimodal regression(`db20d34`)root cause is
STILL unverified after the supposed fix**。

### Tactical(P0.3 prefill warmup directive)

My P0.3 dispatch directive said(`9e964c9` + `8b1a913`):
> "c20b1ce already fixed max_bs to read model.max_concurrent_prefill_requests
> (line 42-43), so DECODE paths are now warmed for batch sizes up to 8"

**This claim was wrong**。Decode paths only warm `0..num_slots`(not 0..8)。
P0.3 prefill warmup pass cannot extend batch sizes beyond num_slots
either(same slot-id constraint)。

**Refined P0.3 scope**:prefill warmup pass that varies **prompt length**
at fixed batch_size=num_slots — different from extending batch sizes。
GEMM shape variation comes from prompt length(M dimension),not batch
count。

### Methodology(SOLID)

This is a **5-layer SOLID gap chain**:

| Layer | Commit | Assumption | Reality |
|-------|--------|-----------|---------|
| 1 | `c20b1ce` | "Extend max_bs to prefill_cap" | Silent loop failure at slot >= num_slots |
| 2 | `db20d34` | "H4 = warmup gap, c20b1ce fixes it" | c20b1ce never actually warmed >num_slots |
| 3 | `1fdd763`(my Phase 0) | "DECODE paths now warmed for 0..8" | Wrong — only 0..num_slots warmed |
| 4 | `c076aae`(audit-of-audit) | Caught my hypothesis-inheritance | But focused on prefill GEMM routing,not slot-id constraint |
| 5 | codex P0.2 work | (Accidental discovery via warmup.rs touch) | Found incoherence |

**§0 first principle in action**:推断 ≠ evidence 全链路。Even multi-layer
audit didn't catch incoherent implementation — needed actual code-level
implementation work to surface the slot-id constraint。

## Validates `d2c2c17` strategic brief

Codex's `d2c2c17` insisted Phase 1 evidence decomposition before P1
commitment because P1 axes might not target the right gap。**This finding
strengthens that argument** — bimodal root cause itself is unverified
because the supposed fix was incoherent。

**Phase 1.A nvtx decomposition** must now ALSO cover:
- Was bimodal regression actually a warmup issue? OR cublasLt heuristic
  cache? OR paged-KV alloc? OR something else?
- Decompose first-burst latency into phases against steady-state
- Don't trust the H4 hypothesis in `db20d34` — c20b1ce didn't fix what
  it claimed to fix

## Recommended next actions

### Pre-Phase-1.A(Claude side,~30 min)

1. Re-run cap=8 fresh-server bench with **revert to `num_slots.min(256)`**(post-codex-P0.2 commit)
2. Compare:if bimodal regression returns,c20b1ce was masking via some
   side-effect。If bimodal stays absent,bimodal was never present
   post-cublasLt-population
3. Output:concrete evidence on whether c20b1ce-era stability was
   warmup-driven OR cublasLt-cache-driven

### Phase 1.A nvtx decomposition(unchanged plan but with new context)

Already-prepped Phase 1.A recipe(`2fafa9e` + `b55bfcd` scoping fix)
remains valid。Decomposing first-burst into phases is the right next
step regardless of whether c20b1ce was working。

### P0.3 directive update

Update P0.3 in pickup queue:
- Removed assumption: "DECODE paths warmed for batch sizes 1..8"
- New assumption:DECODE paths warmed for `0..num_slots`(actual)
- P0.3 prefill warmup pass varies **prompt length M**,fixed batch=num_slots
- LOC unchanged(80-100),scope clarified

## Cross-references

- Discovery commit (codex P0.2 mid-flight):working tree `infer/src/scheduler/cuda/core/warmup.rs`
- Original c20b1ce (incoherent fix):see `git log warmup.rs`
- db20d34 H4 hypothesis:`docs/research/2026-05-08-cap8-default-h4-warmup-cap-rootcause.md`
- 1fdd763 my Phase 0 audit (inherited c20b1ce assumption):pickup queue P0.3 history
- d2c2c17 strategic brief:`docs/research/2026-05-09-eod83-post-b3-strategic-next-axis-roi.md`
- Phase 1.A recipe + scoping fix:`2fafa9e` + `b55bfcd`

## Status

Discovered during codex P0.2 mid-flight implementation。**Bimodal
regression root cause analysis MUST be redone**(c20b1ce-era bench
data may have been measuring something other than what was claimed)。

P0.3 directive scope-clarified;Phase 1.A unchanged but validated as
critical;skill v1.8.0 anti-pattern #22 candidate:**incoherent-fix
masked by silent failure path**(when "fix" loop silent-breaks on
unfulfillable constraint,downstream measures stability incorrectly
attributed to fix)。

## Rule

**Implementation auditing exposes incoherent assumptions that pure
hypothesis-grep cannot**。`b55bfcd` recipe-itself audit caught
scoping bug。This brief catches incoherent-fix assumption。Both
required actually attempting to use the artifact,not just
inspecting its prose。

§0 first principle escalates to:**every fix claim itself must be
license-or-kill verified by trying it,not by trusting the
commit message**。`c20b1ce` would have been kill-able by simply
adding `error!("Warmup loop broke at slot {}", slot); panic!();`
in the silent branch — surface the silent failure。

This is anti-pattern #22 candidate territory for skill v1.8.0
batch trigger。
