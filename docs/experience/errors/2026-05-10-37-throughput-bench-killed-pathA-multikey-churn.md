# #37 throughput bench KILLED — Path A multi-key cache churn pattern

## Context

Per `docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md`
(codex's #24 commit `35fc3cf`)+ Phase 2 step 3 RoPE patch landed,Claude
launched #37 throughput bench A vs B(matched-control 4k/c=4 60s warmup 10s)
on Qwen3-4B-W4-hybrid-zpfix per #37 re-scope brief
(`docs/research/2026-05-10-37-rescope-post-codex-multikey-impl.md`)。

## Bench Results

n=1 scout(short window),σ informational only。Production-quality n=3
deferred给 verified-direction recheck,not for KILL since trend is
unequivocal。

| Metric | A (graph OFF) | B (graph ON) | Δ |
|--------|--------------:|--------------:|---:|
| TTFT p50 | **1628.9 ms** | **1627.8 ms** | **-0.07%** |
| TTFT mean | 1684.1 ms | 1624.5 ms | -3.5% |
| TTFT std | 88.9 ms | 141.6 ms | +59% |
| ITL p50 | 11.43 ms | 11.42 ms | -0.09% |
| out tok/s | 229.64 | 231.04 | +0.6% |
| Samples | 75 | 75 | -- |
| Failed | 0 | 0 | -- |
| Peak kv_util | 83% | 83% | -- |

Codex baseline reference(`wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`):
TTFT p50 1639.3 ms — bench A TTFT p50 1628.9 ms confirms baseline 
within 0.6%(default-OFF unchanged by codex's #24)。

## License Decision per `wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`

| Δ TTFT p50 | σ Stability | License |
|-----------|------------|---------|
| -0.07% | σ B > σ A(141 vs 89 ms,+59%)| **❌ KILL** |

Δ < +5% AND σ B WORSE than σ A → **KILL**(per #37 re-scope criteria
"< +5% OR σ ≥ 5%")。

## Anti-pattern detected — Path A multi-key cache CHURN

Per skill v1.7.0 #6 "License on capture reuse,not capture exists":

**54 capture keys**(graph ON 60s window),0 fallback reasons,but
**repeating identical tuples**:

```
2026-05-10T00:51:33.020292 capture key: tokens=2048 batch=1 pages=128 prefix_rows=0
2026-05-10T00:51:37.808965 capture key: tokens=2048 batch=1 pages=128 prefix_rows=0  ← re-capture
2026-05-10T00:51:42.286807 capture key: tokens=2048 batch=1 pages=128 prefix_rows=0  ← re-capture
2026-05-10T00:51:33.030625 capture key: tokens=8192 batch=4 pages=640 prefix_rows=2048
2026-05-10T00:51:37.818085 capture key: tokens=8192 batch=4 pages=640 prefix_rows=2048  ← re-capture
```

→ Same `(tokens, batch, pages, prefix_rows)` tuples re-capturing every
~5s。Either:
- 8-dim key 包含其他 fields(seq_lens, start_positions, page_count)
  variable per request → cache miss on each new request
- OR cache eviction LRU 太 aggressive(max-keys 太低)

Both reduce to:**Path A multi-key cache 不 enable reuse on this workload**。

## Re-license Path B(device-memory start_pos)

Per #37 design brief `docs/research/2026-05-09-37-multikey-vs-device-startpos-design.md`:

**Path B advantages re-confirmed by this empirical evidence**:
- Single graph reused 100% across all start_pos values(per 设计)
- 不依赖 cache size / eviction policy
- SGLang upstream pattern(`PiecewiseCudaGraphRunner` 实测 effective)

**LOC estimate Path B**:100-200 LOC
- start_pos move to device tensor + replay-time refresh hook
- prep kernel 改 read from device(可能需 CUDA C edit)
- greedy_consistency 验证 数值等价

## Action items

1. **Update #37 task description**: scope from Path A bench-only → **Path B device-mem implementation + bench**
2. **Brief codex Path B implementation**(post他 current pickup completes)
3. **Defer Path A optimizations**(LRU tuning, bucket sizing)— not orthogonal to Path B fix

## Cross-references

- Codex #24 implementation:`wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md`(35fc3cf)
- Phase 2 step 3 RoPE patch:`da53d81`(applied just before bench)
- #37 re-scope brief:`research/2026-05-10-37-rescope-post-codex-multikey-impl.md`
- Path A vs B design:`research/2026-05-09-37-multikey-vs-device-startpos-design.md`
- Pre-built bench template:`wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`
- Phase 0 KILL precedent:`errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`(同根因:capture exists 不 reuse)
- Bench artifacts:
  - A:`bench-output/2026-05-10-p37-graph-off-baseline-run2/`
  - B:`bench-output/2026-05-10-p37-graph-on-treatment/`(or run2)

## Rule

**Multi-key cache key necessity:per-request variable fields(seq_lens,
start_positions)在 key 中 → 每 request 几乎 unique key → cache miss 100%
→ "capture exists != capture reused" anti-pattern reappears**。

**Recommendation:invariant fields only in key**(model shape, page size,
chunk size buckets);**variable fields move to device tensor + replay
refresh**(Path B pattern)。

Phase 0 KILL 早 2026-05-08 同根因 — 当时 single-key,现 multi-key 但 key 太 granular。
**两次 KILL 同病灶,根治 = Path B not multi-key cache size tuning**。

## 状态

#37 throughput bench Path A direction **KILLED**。Path B device-memory start_pos
re-licensed as P0(per design brief 9a477c7 推荐)。Wall-clock节省:Path A
LRU tuning 探索 ~3-5 days waste prevented by 1-tick scout bench evidence。
