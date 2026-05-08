# M_ttft-p99 Phase 1 A/B — `--prefill-max-requests 8` NULL at c=8 4k longctx

> Per `ec7fe9d` Phase 0 confirming H1 mechanism(cap=4 at `qwen3/forward.rs:310`),
> Phase 1 single-variable A/B:run W4A16-zpfix c=8 4k longctx with
> `--prefill-max-requests 8` override vs default(model cap=4)。
>
> **NULL result**:TTFT p50/p99 identical within σ → cap=4 is NOT binding
> at this workload。**6.2× p99/p50 spread observed in `f5cf829` is SPECIFIC
> to W4 c=8 8K agent workload**,not present at c=8 4k longctx random text。

## Phase 1 — Single-variable A/B

**Variable**:`--prefill-max-requests` flag value(None / cap=4 model default vs 8)。

All else identical:
- Model:Qwen3-4B-GPTQ-W4A16-marlin-zpfix(post-zpfix corrected)
- Workload:4k prompt + 256 out,c=8,120s × 10s warmup
- Same hardware sm_89

```bash
# Baseline(default,model cap=4 binding)
./target/release/infer --model-path ... --num-slots 16 --max-seq-len 5120

# Treatment
./target/release/infer ... --prefill-max-requests 8
```

## Results — IDENTICAL within σ

| Metric | Baseline(default cap=4) | **--prefill-max-requests 8** | Δ |
|---|---:|---:|---:|
| TTFT p50 | 4811 ms | **4808 ms** | −0.06%(noise) |
| TTFT p99 | 4886 ms | **4894 ms** | +0.16%(noise) |
| TTFT std | 78.6 ms | **80.5 ms** | +2.4%(noise) |
| ITL p50 | 16.28 ms | **16.28 ms** | 0.00% |
| ITL p99 | 16.43 ms | **16.43 ms** | 0.00% |
| out tok/s | 239.21 | **239.29** | +0.03% |

**Phase 8 verdict**:**NULL within σ < 5% threshold**。Per skill v1.4.0
anti-pattern #13(NULL result is real elimination):**hypothesis "prefill
cap=4 is the TTFT bottleneck at c=8" is REFUTED at this workload**。

## Key insight — the 6.2× spread is workload-shape-specific

`f5cf829` W4 c=8 admission-fix bench:**6.2× p99/p50 spread**(p50 11768
vs p99 72515 ms)。Workload was **W4 agent-tool-resume:8K prompt + 256
resume,c=8 burst**。

This bench:**1.016× p99/p50 spread**(p50 4811 vs p99 4886 ms)。Workload
was **4K random text prompt + 256 output,c=8 sustained**。

**Difference**:
1. **Prompt size**:8K(W4 agent)vs 4K(this run)— **2× more total prefill tokens**
2. **Workload type**:agent burst(`bench_agent_trace.py`)vs sustained random(`bench_guidellm.sh`)
3. **Token budget interaction**:8K × 8 = 64K total prefill at burst,vs 4K × 8 = 32K
   - `max_num_batched_tokens` default 16384 already binding for both,but 8K-prompt forces 4-chunk-per-session vs 1-chunk-per-session at 4K
   - 4-chunk × 8 sessions = 32 prefill steps to drain,vs 8 steps at 4K
   - → 4× more sequential admission cycles at 8K → larger compounding tail

## Updated hypothesis ranking

H1(prefill_max_requests cap=4):**REFUTED at c=8 4k**(this run NULL)
- May still be binding at c=16 W3 if max_num_batched_tokens scales differently

H1'(token_budget × multi-chunk-per-session at 8K):**STRONGER**
- 8K prompt = 2 chunks at 4K chunk size
- 8 sessions × 2 chunks = 16 prefill operations
- token budget 16384 / 4096 = 4 chunks per step
- → 16/4 = **4 step cycles** to drain all 8 sessions
- Predicted TTFT spread:up to 4× → some sessions wait 4 × 4079 ms = 16316 ms
- Combined with chunk-size dynamic adjustments at high pressure,could
  reach 6.2× spread

H4(CUDA Graph batch=5-8 not pre-captured):**still possible**
- Tail sessions hitting batch sizes that need first-encounter compile
- Adds variable ms overhead per session

H2-H3-H5 still in hypothesis space。

## Next steps

### Phase 1.1 — Re-run A/B at W4 8K agent workload
- [ ] Server with `--prefill-max-requests 8` AND larger `--max-num-batched-tokens 32768`
- [ ] Run `bench_agent_trace.py --workload agent-w4-tool-resume --num-concurrent 8`
- [ ] Compare TTFT p99 vs `f5cf829` baseline 72515 ms
- [ ] If p99 drops:H1' confirmed,fix is configuration tuning not substrate change
- [ ] If p99 unchanged:investigate H4 graph capture or H2 KV admission

### Phase 1.2 — `/v1/stats` trace per session timing at c=8 8K
- [ ] Capture `step_phase_us` over time during burst
- [ ] Identify which step does each session enter prefill
- [ ] Verify staircase pattern matches predicted

## Cross-references

- Phase 0 H1 confirmed mechanism: `ec7fe9d`(`docs/research/2026-05-08-ttft-p99-h1-prefill-cap-4-confirmed.md`)
- Plan: `a25416b`
- W4 c=8 admission-fix baseline(8K prompt,p99 72515 ms): `f5cf829`
- This run baseline(4K prompt,c=8): `bench-output/2026-05-08-m_quant-w4a16-zpfix-c8-4k/`
- Cap-of-4 source: `infer/src/model/qwen3/forward.rs:310-320`
- Bench artifact:`bench-output/2026-05-08-m_quant-w4a16-zpfix-c8-4k-prefcap8/`

## Skill v1.4.0 anti-pattern catch

**Wrong-workload investigation trap**:Phase 0 hypothesized H1 from
`f5cf829` 8K agent burst signal,Phase 1 A/B was tested on 4K random
text — different workload entirely。NULL at this workload doesn't
refute H1 at the original workload。

**Rule added(skill v1.4.0)**:**Phase 1 A/B must use the SAME workload
as the Phase 0 empirical signal**。Otherwise NULL is meaningless —
just confirms different workload behaves differently。

This was the methodology gap caught here。Phase 1.1 re-runs A/B at the
correct workload(W4 8K agent c=8)to test H1 at its native shape。

## Status

- ✅ Phase 0 H1 mechanism confirmed via code-grep(`ec7fe9d`)
- ✅ Phase 1 A/B on c=8 4k longctx:NULL(this entry)
- 🔧 Phase 1.1 A/B on c=8 8K W4 agent workload:PENDING(needs proper
     bench setup — Claude actionable next tick)
- ⏳ Phase 0.5 /v1/stats trace:PENDING(codex or Claude)

## Rule

When investigating tail-latency,**workload shape is the variable**,
not just batch size or concurrency。p99/p50 spread depends on:
- Prompt token count(longer = more prefill chunks per session = more
  compounding sequential admission)
- Workload type(burst vs sustained)
- Output token count(longer = more decode amortizes prefill tail)

Always replicate the SAME workload that produced the empirical signal
before A/B-testing fixes。Different workload at "same conc" can have
totally different scheduling behavior。
