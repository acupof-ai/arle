# Cell (a) ≡ Cell (d) post-P0.2 — 4-cell A/B simplifies to 2-cell

> Per `2ba6cbf` cycle stage 25 noting "Cell (a): 🟡 pending recipe(revert
> both,similar pattern)",I realized **cell (a) on current main is
> identical to cell (d)** because P0.2(`232aed5`)already reverted
> c20b1ce as part of its slot-clamp fix。

## Logical observation

Cell definitions(per `3fea979` 4-cell A/B design):
- (a) revert both c20b1ce + 12300c5 → predict 76%
- (b) keep both(historical pre-P0.2 main)→ historical 100%
- (c) keep 12300c5 + revert c20b1ce(current main post-P0.2)→ 100% partial confirmed
- (d) revert 12300c5 + keep c20b1ce → predict 76%

Current main state(per `717b304` grep verified):
- `warmup.rs:36`:`max_bs = num_slots.min(256)` ← c20b1ce REVERTED by P0.2
- `qwen3/forward.rs:321`:`Some(8)` ← 12300c5 KEPT

→ Current main IS cell (c)。

To run cell (a) on current main:
- "Revert both" = revert c20b1ce + revert 12300c5
- But c20b1ce **is ALREADY reverted** in current main(P0.2 did this)
- So "revert both" → simply revert 12300c5

To run cell (d) on current main:
- "Revert 12300c5 + keep c20b1ce" = revert 12300c5 + (keep current c20b1ce-revert state)
- But c20b1ce-revert IS the current state,so "keep c20b1ce" means keep CURRENT state(which has c20b1ce reverted)
- So "revert 12300c5 only" → simply revert 12300c5

→ **Cell (a) ≡ Cell (d) on current main**。Single experiment yields both。

## Implication for tomorrow's pickup

Per `1bf408d` Cell (d) recipe(~30 min wall-clock,sed Some(8)→Some(4)):
- Single experiment now serves as BOTH cells (a) and (d)
- Decision matrix simplified:
  - **TTFT ≥ 300 ms / turn ~76%**:12300c5 confirmed real fix(both
    semantics consistent — c20b1ce contribution = 0%,12300c5 = 100%)
  - **TTFT ≤ 250 ms / turn ~100%**:12300c5 also no-op,bimodal mitigation
    came from elsewhere(major surprise per `1bf408d`)

## 4-cell A/B reduces to 2-cell on current main

| Cell | Pre-P0.2 design | Post-P0.2 reality | Action |
|------|-----------------|-------------------|--------|
| (a) | revert both | revert 12300c5(c20b1ce already reverted)| same as (d) |
| (b) | keep both | non-reproducible(c20b1ce no longer kept after P0.2) | historical only |
| (c) | revert c20b1ce | current main | already PASS partial(c=1)|
| (d) | revert 12300c5 | revert 12300c5 | run experiment |

**Net empirical work**:**1 experiment**(`1bf408d` recipe)to confirm 12300c5
attribution。Cell (a) recipe = Cell (d) recipe = same。

This means tomorrow's pickup has **even less friction**:single ~30 min
sed-build-bench-restore experiment closes the H7-A audit chain。

## Caveat — cell (b) is historical

Cell (b) "keep both" is no longer reproducible **post-P0.2** because P0.2
permanently removed c20b1ce's `max_bs.max(prefill_cap)` extension。Cell (b)
empirical data exists ONLY in pre-P0.2 wins entries(`b85929b` LICENSE bench
at 241 ms / 100% turn success on warm-cache shared-prefix burst)。

For cell (b) re-test,one would need to manually re-apply c20b1ce on top of
P0.2 — non-trivial because P0.2 simplified the warmup.rs structure。Not
recommended unless cell (d) shows surprising result requiring investigation。

## §0 first principle observation

This finding emerged from §0 self-audit:**before writing cell (a) recipe,
verify cell (a) is actually different from cell (d)**。Found they're identical
on current main。Saved tomorrow's pickup ~30 min(otherwise would write
duplicate recipe and run duplicate experiment)。

The pattern:**before producing artifact X,check whether X already exists
or reduces to existing artifact Y**。Anti-pattern #24 candidate?

## Skill v1.8.0 anti-pattern #24 candidate

> **Anti-pattern #24**:Cell-collapse blindness。When designing controlled
> experiments(N-cell A/B),verify cell INDEPENDENCE under current substrate
> state。If 2 cells differ only in dimensions that current main has already
> normalized(e.g. P0.2 reverted c20b1ce → cells (a) and (d) both "revert
> 12300c5 only"),they collapse to single experiment。Cure:after each
> substrate landing,re-derive each cell's required edits and check for
> identity overlaps。

This complements skill v1.8.0 batch:
- #20 hypothesis-inheritance(c076aae)
- #21 recipe-itself audit(b55bfcd)
- #22 twin-commit attribution(3fea979)
- #23 truncated-output partial-view(156d2c2)
- **#24 cell-collapse blindness**(this brief)

5 candidates ready when v1.8.0 batch triggers。

## Cross-references

- `1bf408d` Cell (d) recipe(now also cell (a) recipe per this brief)
- `717b304` Current main = cell (c) grep verification
- `2ba6cbf` cell-by-cell pickup queue table
- `3fea979` 4-cell A/B original design
- `232aed5` P0.2 LANDED with c20b1ce revert built-in
- `bbedbc9` Layer-8 num_slots=8 gate
- §0 first principle:CLAUDE.md "求真务实,追求极致"

## Status

Tomorrow's experimental work simplified from 4 cells to 1 experiment
(serves as both (a) and (d))。`1bf408d` recipe is fully ready。
Cell-by-cell pickup queue table at `2ba6cbf` accurately reflects
current state — Cell (a) "pending recipe" is now resolved as
"identical to cell (d) recipe per eod99 collapse finding"。

**Tomorrow's actual experimental work**:**1 experiment**,~30 min wall-
clock,**both** cells(a)+(d)results from single run。

§0 in action:before pre-staging more recipes,check whether existing
recipes already cover the case。Saved 30 min + reduced cognitive load
on tomorrow's pickup。
