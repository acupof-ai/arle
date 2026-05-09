---
title: M_warmup-prefill-pass-directive vs current warmup.rs — Pass 2 naming collision + reset_for_warmup_clear gap confirmed
date: 2026-05-10
type: research
status: m-warmup-directive-needs-minor-revision-before-pickup
---

# M_warmup-prefill-pass-directive vs current warmup.rs — Pass 2 naming collision + reset_for_warmup_clear gap confirmed

> Pre-pickup reconciliation for Task #35 (cap=8 prefill pre-warm fix
> per Step 2.B'). M_warmup-prefill-pass-directive.md was written
> 2026-05-08 (last commit `56dbd1c`); reconciled against current
> `warmup.rs` THIS tick reveals 2 small adjustments needed before
> codex can directly apply the directive.

## §0 Direct evidence (raw grep on warmup.rs THIS tick)

### Pass 2 naming collision

```bash
$ grep -nE 'Pass [0-9]|prefill warmup' /home/ckl/projects/arle/infer/src/scheduler/cuda/core/warmup.rs
23:    /// 3. Pass 2 (graph-capture mode only) re-captures graphs with the autotuned
150:                    // Pass 2: re-capture with autotuned algorithms.
```

**Current warmup.rs ALREADY has a "Pass 2"** — for graph-capture
re-capture with autotuned algorithms. M_warmup directive proposes
adding "Pass 2" for prefill warmup → naming collision.

**Resolution**: directive should rename the new pass to "Pass 3" OR
restructure existing Pass 2 to combine prefill warmup + autotuned
re-capture. Recommend "Pass 3" for minimal-edit clarity.

### forward_prefill_batch_with_pool existence verified

```bash
$ grep -rln "forward_prefill_batch_with_pool" /home/ckl/projects/arle/infer/src/model/
/home/ckl/projects/arle/infer/src/model/qwen3/forward.rs
/home/ckl/projects/arle/infer/src/model/qwen35/forward.rs
```

✅ M_warmup directive's Step 1 reuse-existing-infrastructure plan
intact — `forward_prefill_batch_with_pool` exists in both qwen3 and
qwen35 model implementations.

### reset_for_warmup_clear NOT in tree

```bash
$ grep -rln "reset_for_warmup_clear" /home/ckl/projects/arle/infer/src/
(no output)
```

✅ M_warmup directive Step 3 anticipated this: "May need
`reset_for_warmup_clear` method added to State trait if not present"
— confirmed needs adding. Per directive: "check current pattern + replicate"
the post-decode-warmup reset already in warmup.rs.

### existing reset patterns in warmup.rs

```bash
$ grep -nE 'reset|alloc_tokens|free_slot|paged_kv_pool' /home/ckl/projects/arle/infer/src/scheduler/cuda/core/warmup.rs | head -10
70:                if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
164:                    self.paged_kv_pool.free_slot(slot);
193:            let page_size = self.paged_kv_pool.page_size;
245:                Some(&mut self.paged_kv_pool),
```

Existing pattern uses `paged_kv_pool.alloc_tokens` + `free_slot` for
warmup KV management. M_warmup directive's `reset_for_warmup_clear`
would extend this pattern to handle prefill warmup KV cleanup.

## §1 Recommended directive adjustments (small, ~5-10 LOC delta)

### Adjustment 1 — rename "Pass 2" → "Pass 3" in directive

In `M_warmup-prefill-pass-directive.md` Step 2 section, change:
```diff
-### Step 2 — Add prefill warmup pass to `warmup.rs`
+### Step 2 — Add Pass 3 (prefill warmup) to `warmup.rs` (avoid existing
+###          Pass 2 graph re-capture name)
```

And in the rust code snippet:
```diff
-// Pass 2 (per 641e9bf): prefill warmup. Drive forward_prefill_batch
+// Pass 3 (per 641e9bf): prefill warmup. Drive forward_prefill_batch
```

### Adjustment 2 — confirm reset_for_warmup_clear pattern

Step 3 already says "May need `reset_for_warmup_clear` method added".
Confirm by referencing this entry: ARLE has no such method THIS tick,
needs implementing per existing `paged_kv_pool.free_slot` + state
reset pattern in warmup.rs:163-165.

## §2 Updated Task #35 scope estimate

Original directive estimate: ~100-150 LOC, 1 day codex.

Reconciled scope:
- Add Pass 3 (prefill warmup) to warmup.rs: ~50-70 LOC
- Add `reset_for_warmup_clear` method to State trait + impls (qwen3,
  qwen35, deepseek): ~30-50 LOC  
- Add tests + bench validation: ~20-40 LOC
- Total: ~100-160 LOC + 1-1.5 days codex (per 56dbd1c estimate)

No major surprises. Directive is well-founded; just needs the Pass 2
→ Pass 3 rename + explicit acknowledgment of `reset_for_warmup_clear`
addition.

## §3 Cross-references

- M_warmup-prefill-pass-directive.md (`56dbd1c`, 2026-05-08, last touched)
- Task #35 (cap=8 prefill pre-warm fix, codex own, ~100-150 LOC scope)
- 641e9bf (cap=8 bimodal trigger CONFIRMED — origin of M_warmup directive)
- warmup.rs:23+150 (existing Pass 2 = graph re-capture, naming collision)
- qwen3/forward.rs + qwen35/forward.rs (forward_prefill_batch_with_pool reuse)
- Pickup queue per `2c736d0`/`9ccd36b` next-session pickup state

## §4 Status

M_warmup directive verified mostly accurate vs current code state.
Two small adjustments needed before codex pickup:
1. Rename proposed "Pass 2" to "Pass 3" (avoid naming collision)
2. Acknowledge `reset_for_warmup_clear` confirmed missing, needs adding

These adjustments could be applied to `M_warmup-prefill-pass-directive.md`
directly OR cited from this research entry when codex picks up #35.

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(warmup.rs grep + qwen3/qwen35 forward.rs grep + git log on directive
file — all THIS tick).
