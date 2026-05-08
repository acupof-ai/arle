# Current main IS cell (c) of 4-cell A/B — partial H7-A evidence

> Per `bb888cb` 4-cell A/B distinguishing experiment codified for c20b1ce
> dead-code confirmation。Post `232aed5`(P0.2 LANDED),current main state
> already aligns with cell (c)。Code-grep verification + existing bench
> data = partial H7-A evidence already collected。

## Cell mapping(post `232aed5` LANDED)

| cell | revert c20b1ce | revert 12300c5 | predicted | current main? |
|------|:------:|:------:|:---:|:---:|
| (a) | YES | YES | 76% | ❌ — needs revert experiment |
| (b) | NO | NO | 100% | ❌ — was prior main pre-`232aed5` |
| **(c)** | **YES** | **NO** | **100%** | **✅ CURRENT MAIN** |
| (d) | NO | YES | 76% | ❌ — needs revert experiment |

## Current main state — code-grep verification

`warmup.rs:36`(post `232aed5`):
```rust
let max_bs = num_slots.min(256);
```
↑ c20b1ce REVERTED(was `num_slots.max(prefill_cap).min(256)`)。

`qwen3/forward.rs:321`:
```rust
Some(8)
```
↑ 12300c5 KEPT(was `Some(4)` pre-12300c5)。

→ Current main = **cell (c) state exactly**。No additional setup needed
to bench cell (c) — it's already main。

## Cell (c) partial empirical evidence(from `232aed5` bench)

`hybrid-phase1b-loader-regression` bench in P0.2 wins entry(`232aed5`):
- Profile:512-in / 64-out / c=1
- TTFT p50:68.4 ms
- ITL p50:14.02 ms
- 0 errors,full completion

→ Cell (c) at c=1 short prompt:**PASS**,0 errors。

⚠ **Caveat**:P0.2 bench was c=1 short-prompt,not the W3 c=4 cap=8
multi-turn workload that originally showed 76→100% improvement
attributed to c20b1ce。**For full H7-A confirmation at the workload
that revealed bimodal regression,need W3 c=4 cap=8 fresh-server bench
on current main**。

If that bench shows 100% turn success → cell (c) ≈ cell (b) prediction
validated → c20b1ce confirmed dead code → 12300c5 alone is the actual
fix。

If <100% → c20b1ce had non-trivial effect → 7-layer audit conclusions
need refinement。

## Cell (b) historical empirical evidence

Pre-`232aed5` main(both c20b1ce + 12300c5 active):
- Wins entry `2026-05-08-w3-c4-cap8-default-clean-100pct-tt-improved.md`
  reported 100% turn success at W3 c=4 cap=8(now annotated per
  Layer-8 confound — bench had `--num-slots 4` baseline vs `--num-slots
  16` post-run mismatch per `655accf`)
- Bench attribution to c20b1ce + 12300c5 jointly = wrong-attribution
  per Layer-7 closure(`3fea979`):c20b1ce was NO-OP,12300c5 was actual fix

→ Cell (b) historical data **with attribution corrected**:100% turn
success on production-default codepath came from 12300c5,not c20b1ce。

## Cells (a) and (d) — remaining experiments

**Cell (a)**:revert BOTH c20b1ce + 12300c5(`Some(4)`)→ predicted 76%
- Effort:`git revert 232aed5 12300c5`(commit pair)+ build + W3 c=4 cap=8 bench
- ~30 min wall-clock
- If 76% confirmed → re-validates pre-12300c5 baseline

**Cell (d)**:keep current main + revert 12300c5 → predicted 76%
- Effort:patch `qwen3/forward.rs:321` `Some(8) → Some(4)` + build + bench
- ~30 min wall-clock
- If 76% confirmed → c20b1ce dead-code irrelevant when 12300c5 is the
  real fix

→ Cell (d) is the **single most informative experiment**:isolates 12300c5
contribution alone with c20b1ce-revert held constant。

## Recommended tomorrow's pickup ordering

1. **Verify cell (c) at W3 c=4 cap=8**(~10 min Claude bench):
   confirm current main shows 100% turn success at the workload that
   revealed bimodal regression
2. **Cell (d) experiment**(~30 min):patch `Some(8)→Some(4)`,build,
   bench W3 c=4 cap=8 → if 76% confirmed,12300c5 alone is the fix
3. **Cell (a) experiment**(~30 min):revert 232aed5 too,bench → 76%
   reconfirms baseline degraded path
4. **Decision**:if all predictions confirmed,revert c20b1ce dead code
   from main per "no half-states" rule + skill v1.8.0 #22 codification

Total Claude effort:~70 min wall-clock for all 4 cells。

## Layer-8 num_slots gate(per `655accf`)

⚠ **CRITICAL**:all 4 cells MUST run with `--num-slots 8` constant。
The original "validation bench" had multi-variable confound
(`bwa4piqqx` num_slots=4 vs `b1mm1k0r7` num_slots=16)。Repeating that
trap would invalidate the 4-cell A/B evidence。

Bench command template:
```bash
./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 8 \  # Layer-8 invariant
  --max-seq-len 5120
# Then:
scripts/bench_agent_trace.py --workload agent-w3-short-multiturn \
  --num-concurrent 4
```

## Cycle-completion observation

This brief is **stage 24** of bidirectional audit cycle — Claude
codifies "current main is cell (c)" by code-grep verification post
codex's P0.2 LANDED(stage 23)。Cycle continues to compound even
post-substrate-LANDING:partial empirical evidence already gathered,
remaining cells pre-staged for tomorrow's pickup with explicit
commands + Layer-8 num_slots gate。

## Cross-references

- 4-cell A/B framing:`bb888cb`(pickup queue)+ `3fea979`(layer-7 closure)
- Layer-8 num_slots gate:`655accf`(empirical comparison must control all variables)
- P0.2 LANDED:`232aed5` `feat(cuda): load hybrid W4 Marlin side tensors`
- Pre-`232aed5` cell (b) annotation:`655accf`(2 wins entries) + `9bc4729`(3rd entry)
- Cycle stages 1-23:pickup queue table

## Status

Cell (c) state code-grep verified ✅;c=1 partial bench evidence from
`232aed5` ✅。Tomorrow's pickup:run W3 c=4 cap=8 cell (c) bench
(~10 min) → cell (d) experiment(~30 min) → optional cell (a)
verification → c20b1ce dead-code revert decision。

§0 first principle:推断 ≠ evidence,但 partial-evidence 也 codify 起来,
为 tomorrow's full-evidence collection 减少 setup cost。
