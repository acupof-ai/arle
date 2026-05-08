# cap=8 default-vs-override turn-success variance — investigation needed before further action

> Per `150b4c4` post-`12300c5` cap=8 default flip verification:
> - TTFT improvement holds(-85% p99 vs cap=4 baseline)✅
> - **Turn success regression**:194/256(76%)vs `19d12c2` override 257/257(100%)❌
>
> Same nominal cap=8,different outcomes。Single-variable A/B violated
> at production-readiness level。Per §0 SOLID,investigate before
> pushing further changes。

## Empirical comparison

W4 c=8 8K agent burst,3 different runs:
| Run | Cap | Turn success | TTFT p99 | Peak mem | Wall total |
|-----|-----|-------------:|---------:|---------:|---------:|
| `f5cf829` | 4(default at time)| 256/256 | 72515 ms | similar | ? |
| `19d12c2` | 8(`--prefill-max-requests 8` override)| 257/257 | 10259 ms | 15272 MB | ~860 s |
| `150b4c4` | 8(default after my `12300c5` flip)| **194/256** | 11182 ms | **15880 MB** | **2290 s** |

TTFT win HOLDS across both cap=8 runs(p99 ~10-11k vs cap=4 72k)。
Turn success FAIL on default-cap=8 only。

## Variance suspects

### H1 — Memory pressure(strongest)
- Default cap=8:peak **15880 MB** vs override cap=8:**15272 MB** = +608 MB
- 16GB GPU → only **120 MB headroom**(15880 / 16000 = 99.2%)
- vs override:**728 MB headroom**(15272 / 16000 = 95.5%)
- Plausible:tighter memory → some sessions hit OOM/timeout

Why default would have +608 MB?
- Same code path,same kernel,same workload
- Possible:fresh build cold-start triggers larger CUDA context init
- Or:cumulative allocator fragmentation from longer wall time(2290 s vs 860 s = 2.7×)
- Or:KV pool fragmentation accumulates over more sessions / turns

### H2 — Run-to-run variance(natural)
Wall total 2.7× longer = more pressure accumulating。If turn success rate
drops 10% per hour due to KV fragmentation / reservation churn,2290 s
run vs 860 s would expect ~6% extra failures from fragmentation alone。

But 76% vs 100% = 24% gap,much bigger than 6%。Variance explains a portion,
not all。

### H3 — Cold-start graph capture tax
First-encounter batch sizes 5-8 trigger CUDA graph capture(if any path
captures graphs)。Override test maybe ran AFTER warmup with cap=4 ran
before;default test ran cold。

But ARLE doesn't capture prefill graphs(M_pf-graph KILLED)。Decode
graphs may capture per-batch-size first encounter。Plausible but small
contribution。

### H4 — Build artifact difference
`12300c5` was 1-line edit。If override test ran on a separate build with
some other latent change(eg incremental rebuild artifacts),could affect
behavior。

Mitigation:`cargo clean && cargo build --release` from clean source
before re-test。

## Investigation plan

### Step 1 — Rerun override test fresh
Re-run `--prefill-max-requests 8` override on a NEW infer instance(no
warmup,fresh process)to baseline override-only at default startup:
```bash
cargo clean && cargo build --release -p infer --features cuda
./target/release/infer ... # serve fresh
# Run W4 c=8 8K spec same as 19d12c2
```

If override test ALSO drops to 76% turn success → confirms variance is
NON-cap factor(memory / cold start / fragmentation)。Cap=8 itself OK。

If override test still 100% → confirms my `12300c5` code change indirectly
broke something(should not — 1-line edit can't,but verify)。

### Step 2 — Memory pressure isolation
Run cap=8 default with `--max-seq-len 6144` instead of 9216:
```bash
./target/release/infer ... --max-seq-len 6144  # reduce KV pool pressure
```

Expected:peak mem drops by ~30%,turn success returns to 100% if H1 is
real。

### Step 3 — KV fragmentation isolation
Run cap=8 default but restart server after every 64 turns(scripted)。
Expected:if H1+H2 are real,frequent restart prevents fragmentation
accumulation,turn success returns to 100%。

## Decision tree

| Step 1 result | Step 2 result | Step 3 result | Verdict |
|---|---|---|---|
| Override drops to 76% | n/a | n/a | Variance only,my flip OK,re-bench with controlled conditions |
| Override 100% | success drops with shorter ctx | n/a | Memory pressure real,my flip may need adaptive logic |
| Override 100% | no change | success drops | KV fragmentation,orthogonal scheduler issue |
| Override 100% | no change | no change | Code change broke something — REVERT and investigate |

## Risk assessment

If turn success regression is a REAL issue at default cap=8,my
`12300c5` flip could cause **76% production failure rate** for users
serving W4A16/W4A8 via Qwen3 Marlin with c=8 8K agent workload。

Mitigation options:
- A. **Revert `12300c5`** until variance investigated → keeps cap=4 default
- B. **Make cap conditional on memory budget**(e.g. cap=8 only when
  `--max-seq-len ≤ 6144`,else cap=4)
- C. **Investigate first**(Step 1)and decide。If variance(Step 1 PASS),
  no change needed。

Recommendation:**C investigate first**(Step 1 = ~5 min GPU)。If
variance confirmed,no action。If not,revert(A)。

## Strategic context

`12300c5` flip was applied based on:
- `19d12c2` override test 257/257(LICENSED)
- `27fd5de` cross-shape verify(W3 c=16 384/384,W4 c=8 256+1/256)

The cross-shape data was strong。`150b4c4` is one new datapoint that
contradicts。Per Bayesian update:single contrarian datapoint should
NOT immediately revert,but should trigger investigation。

Per §0 SOLID first principle:
- 解决 root cause(`150b4c4` variance source)
- 实证 evidence(Step 1 rerun)before pushing more changes
- 不 silent 放过(this brief documents the gap)

## Cross-references

- `12300c5` cap=8 flip applied(my edit)
- `19d12c2` cap=8 override LICENSED 257/257
- `27fd5de` cap=8 multi-shape verify
- `150b4c4` cap=8 default 194/256 regression
- `f5cf829` cap=4 baseline 256/256
- `b708e00` original admission deadlock fix(introduced cap=4)

## Status

**Investigation OPEN**。Next concrete action:Step 1 rerun override test
fresh on current `12300c5` build to isolate variance from code-change
side effects。Estimated 30 min GPU(rebuild + serve + bench)。

Codex(or user direct via shell)can execute:
```bash
git status  # confirm on 12300c5
cargo clean
cargo build --release -p infer --features cuda
RUST_LOG=info ./target/release/infer --model-path ... --port 8000 ... &
python scripts/bench_agent_trace.py --workload w4-c8-8k ... --prefill-max-requests 8
```

If override on current build still 100% → my edit is fine,150b4c4 was variance。
If override ALSO regresses → code change broke something despite 1-line。

**This brief blocks further cap-related changes** until investigation
resolves。

## Methodology rule

Per §0 SOLID:single contrarian empirical datapoint(`150b4c4` 76%)
should not auto-trigger revert,but **MUST trigger SOLID investigation
before pushing further changes**。Burst of validating commits +
production-applied flip without follow-up verify violates "推断 ≠ SOLID"
principle。
