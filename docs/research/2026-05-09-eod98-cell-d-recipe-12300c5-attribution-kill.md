# Cell (d) experiment recipe — 12300c5 attribution kill criterion

> Per `717b304` confirmation that current main = cell (c) of 4-cell A/B,
> the **most informative single remaining experiment is cell (d)**:revert
> 12300c5 to Some(4) on current main,bench W3 c=4 cap=8 with **fixed
> num_slots=8**(per Layer-8 confound gate `bbedbc9`)。
>
> Predicted result:**76% turn success**(reverting cap=8→4 admission)。
> If observed → 12300c5 is the actual fix(closes 7-layer H7-A definitively)。
> If 100% → 12300c5 contribution = 0%,bimodal must come from other axis。

## Code grep verification(2026-05-09 EOD+98)

Confirmed current main state:
- `infer/src/scheduler/cuda/core/warmup.rs:36`:
  `let max_bs = num_slots.min(256);` ✓ (c20b1ce reverted by P0.2 232aed5)
- `infer/src/model/qwen3/forward.rs:321`:`Some(8)` ✓ (12300c5 KEPT)

→ Cell (c) state confirmed。

## Cell (d) experiment recipe(~30 min wall-clock)

### Step 1 — Revert 12300c5 in-place(2 min)

```bash
# Edit infer/src/model/qwen3/forward.rs:321
sed -i 's/Some(8)/Some(4)/' infer/src/model/qwen3/forward.rs
# Verify
grep -E "Some\([0-9]+\)" infer/src/model/qwen3/forward.rs | head -3
```

Expected:line 321 now shows `Some(4)`。

### Step 2 — Build release CUDA(~5 min)

```bash
CUDA_HOME=/opt/cuda \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo build --release -p infer --features cuda
```

### Step 3 — Server with **fixed num_slots=8**(critical Layer-8 gate)

```bash
./target/release/infer \
    --model-path infer/models/Qwen3-4B \
    --port 8000 \
    --num-slots 8 \
    --max-seq-len 8192 &
SERVER_PID=$!
sleep 5
```

**CRITICAL**:`--num-slots 8` MUST match cell (c) bench condition(per `bbedbc9`
Layer-8 num_slots gate)。Layer-8 confound was caused by num_slots changing
between baseline and treatment。Cell (d) must hold num_slots constant。

### Step 4 — Bench W3 c=4 cap=8 fresh-server(~10 min)

Use the canonical multitenant burst bench(matches `b85929b` LICENSE
protocol):

```bash
/home/ckl/projects/arle/.venv/bin/python \
    scripts/bench_multitenant_burst.py \
    http://localhost:8000 Qwen/Qwen3-4B
```

Run N=5 paired(per `3c334ef` LICENSE protocol),capture TTFT p50 +
turn success per run。

### Step 5 — Compare to current main(cell c)numbers

| Run | Cell | Cap | TTFT p50(median of N=5)| Turn success |
|-----|------|----:|-------------------------:|-------------:|
| `b85929b` LICENSE | (c)| 8 default | 241 ms | 100% |
| **(d) 12300c5 reverted** | (d)| **4** default | **predict ~318 ms / 76%** | tbd |

### Step 6 — Restore 12300c5(2 min)

```bash
kill $SERVER_PID
sed -i 's/Some(4)/Some(8)/' infer/src/model/qwen3/forward.rs
git diff infer/src/model/qwen3/forward.rs  # verify clean revert
```

## Decision matrix

| (d) result | 12300c5 attribution | Action |
|------------|---------------------|--------|
| **TTFT p50 ≥ 300 ms / turn success ~76%** | **Confirmed real fix** | Skill v1.8.0 anti-pattern #22(twin-commit attribution)closes definitively。Document attribution in `wins/2026-05-09-cell-d-confirms-12300c5.md` |
| **TTFT p50 ≤ 250 ms / turn success ~100%** | **12300c5 also no-op** | Major surprise — bimodal mitigation came from neither c20b1ce nor 12300c5。Investigate other concurrent changes(b3c2ed0,etc.)|
| **Mixed(p50 ~270 ms,turn 88%)** | Partial | 12300c5 is contributor but not sole cause。Investigate co-shipping changes |

## §0 SOLID gates

Per CLAUDE.md §0 first principle:
1. **No PushNotification needed during experiment**:codex is idle awaiting
   user。Tomorrow's Claude/codex picks up directly。
2. **Confound isolation**:`--num-slots 8` MUST be constant per `bbedbc9`。
3. **N=5 paired protocol**:matches `3c334ef` LICENSE precedent。
4. **Wins entry mandatory** post-experiment per CLAUDE.md "MANDATORY —
   every runtime change produces a bench entry"。

## Skill v1.8.0 anti-pattern #22 — empirical validation gate

This experiment is the **empirical validation step** for anti-pattern #22
(twin-commit fix attribution trap)。If(d)confirms predicted ~76%:
- Anti-pattern #22 transitions from **candidate** to **empirically-
  grounded codified rule**
- Skill v1.8.0 batch ready to land(#20 + #21 + #22 + #23 all evidenced)
- Future twin-commit "fix" claims must include **revert each in turn,
  measure individual contribution** as license criterion

If(d)reveals 12300c5 also no-op,anti-pattern #22 framing needs revision
(12300c5 wasn't the real fix either,need to find what actually was)。

## Cross-references

- `717b304` current main IS cell (c)(grep verified `warmup.rs:36` +
  `qwen3/forward.rs:321`)
- `3fea979` Layer-7 closure(12300c5 was actual fix hypothesis)
- `bbedbc9` Layer-8 num_slots gate(critical fixed-config requirement)
- `655accf` 2 wins entries annotated with corrected attribution
- `9bc4729` 3rd doc annotation(downstream-citing-document scan complete)
- `b85929b` LICENSE bench(241 ms TTFT p50 cell (c)reference data)
- `c20b1ce` reverted by P0.2 `232aed5`
- `12300c5` cap=4→8 admission flip
- `bench_multitenant_burst.py` canonical bench
- §0 first principle:CLAUDE.md "求真务实,追求极致"

## Status

Cell (d) recipe **copy-paste-ready** for tomorrow's pickup。~30 min
wall-clock,single-experiment definitive answer on 12300c5 attribution。

This brief is **decision-supporting,NOT decision-making** — tomorrow's
pickup chooses whether to run cell (d) before P0.0 Phase 1.A or after。
Both paths valid;cell (d) provides closure on 7-layer audit chain,
P0.0 Phase 1.A provides multi-tenant TTFT decomposition for P1 axis
selection。

§0 in action:partial-evidence already documented(`717b304`),this brief
adds full-evidence recipe to convert "current main = cell (c)" hypothesis
into "12300c5 is real fix"empirical claim。
