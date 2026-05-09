---
title: Post-#37 license decision tree — fast next-axis selection
date: 2026-05-10
type: plan
status: ready-for-execution
---

# Post-#37 license decision tree — what to do based on bench A/B Δ

> Codex's Path B impl pre-commit evidence(per `c021053`)is HIGH confidence
> for predicted TTFT improvement。This brief defines **explicit Claude
> action** for each license outcome,enables fast forward momentum
> post-codex-commit。

## Bench A/B execution(post codex commit landing)

```bash
# Single command runs full pipeline:
./scripts/post_p24_commit_pipeline.sh full

# OR manual A/B if pipeline misbehaves:
# 1. Build release
env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
   INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
   TORCH_CUDA_ARCH_LIST=8.9 \
   cargo build --release -p infer --features cuda

# 2. Bench A: graph OFF baseline
INFER_HYBRID_W4A8_PREFILL=1 \
   ./target/release/infer --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
   --port 8765 --num-slots 8 --max-seq-len 8192 --admission-policy prefix-aware &
sleep 40
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
   scripts/bench_guidellm.sh p37-pathB-graph-off-baseline \
   --target http://127.0.0.1:8765 --model Qwen3-4B-W4-hybrid-zpfix \
   --concurrencies 4 --max-seconds 60 --warmup 10 \
   --data 'prompt_tokens=4096,...,output_tokens=256,...'
killall infer

# 3. Bench B: graph ON treatment
INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1 \
   ./target/release/infer ... &
sleep 40
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
   scripts/bench_guidellm.sh p37-pathB-graph-on-treatment \
   --target http://127.0.0.1:8765 --model Qwen3-4B-W4-hybrid-zpfix \
   --concurrencies 4 --max-seconds 60 --warmup 10 \
   --data 'prompt_tokens=4096,...,output_tokens=256,...'
killall infer
```

Reference baselines:
- Codex baseline(2026-05-09 reverify):**TTFT p50 1639.3 ms**(Δ +76.6% slower than SGLang 928.4 ms)
- Path A KILL bench(2026-05-10):TTFT p50 1628.9 ms(no improvement)
- SGLang reference:**TTFT p50 928.4 ms**(target)

## License decision tree

### Tier 1 — Strong proceed(Δ ≥ +25%)

**Outcome**:TTFT 4k/c=4 ≤ 1230 ms(close ≥ 25% of gap to SGLang 928 ms)

**Claude action**(immediate):
1. Fill `wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`(`1168381`)→ rename without TEMPLATE prefix
2. Mark task #37 **completed**
3. Update `docs/index.md` "Last refreshed" + `CHANGELOG.md` "Latest Updates" with #37 wins
4. Brief codex on **next P0 axis** — recommend `#36 PrefixAwareAdmission`(close SGLang multi-tenant 2× gap),OR architectural pivot(multi-GPU,model-tier,distributed)

### Tier 2 — Proceed(Δ +10-25%)

**Outcome**:TTFT 4k/c=4 in 1230-1475 ms range(close 10-25% of gap)

**Claude action**:
1. Fill wins template + commit + push
2. Mark #37 completed
3. Brief codex on **incremental axis stacking** — Path B Phase 2 optimizations(invariant-fields-only key,if cache hit < 100% in this Tier 2)
4. Sub-axis:8-key LRU → 16-key cache(higher hit rate at burst load)
5. Then move to #36 OR #30 Hybrid W4A16/W4A8 dispatch

### Tier 3 — Marginal(Δ +5-10%)

**Outcome**:TTFT 4k/c=4 in 1475-1557 ms range(modest improvement)

**Claude action**:
1. Wins entry NOT a "wins" — write **research entry** documenting partial improvement + cache hit rate analysis
2. Mark #37 still in_progress with "partial — Tier 3" status
3. Investigate cache hit rate counter — if < 80%,Path B's LRU eviction may be too aggressive for c=4 workload。Codex pickup:增 LRU size to 16-32
4. Or pivot to **architectural axis**(scheduler / continuous batching mods)

### Tier 4 — KILL(Δ < +5% OR cache hit rate < 50%)

**Outcome**:TTFT essentially unchanged from baseline(like Path A KILL Δ -0.07%)

**Claude action**:
1. **Immediate KILL errors entry**(per `e462c53` template):
   - Cite Path B impl evidence(7-dim brief match,smoke LRU reuse confirmed,kv_last_page_len fix)
   - Identify why empirical Δ failed despite functional gates PASS
   - Hypothesize root cause(launch overhead not binding constraint?KV transfer dominant?metadata refresh overhead eats win?)
2. Mark task #37 **completed with KILL note**(implementation work done,bench result negative)
3. Critical finding:**`launch overhead is not the binding constraint for c=4 4k/256 W4-hybrid prefill on 4070 Ti SUPER`** — SGLang gap is in different axis(possibly attention compute,scheduler queue management,or kernel-internal optimization)
4. Pivot:
   - `#30 Hybrid W4A16/W4A8 dispatch` — change quantization split per phase
   - `#36 PrefixAwareAdmission` — close SGLang multi-tenant 2× gap(different axis)
   - **Re-baseline against SGLang on different workload**(8k,16k context to test attention compute scaling)to find true binding axis

## Anti-pattern check(both wins AND KILL paths)

Per skill v1.7.0 #6:always verify reuse vs capture:
- `cudaGraphLaunch` count vs `cudaGraphInstantiate` count(via nsys cuda_api_sum)
- Server log capture key count vs request count
- For Tier 1/2:expect `launch >> instantiate`,`captures < requests`(reuse working)
- For Tier 3/4:may show `launch ~ instantiate`(re-capture pattern)— corroborates KILL

## Workflow time budget

- Bench A 60s + warmup 10s + boot 35s = **2 min per bench**
- N=3 each = **12 min total**
- Plus tear down + analysis = **20 min full bench cycle**
- License decision + Tier 1/2/3/4 doc + commit = **5-10 min**
- **Total: ~30 min** post codex commit landing

## Cross-references

- Codex Path B draft wins:`docs/experience/wins/2026-05-10-bench-37-pathb-device-metadata-prefill-graph.md`(untracked,pre-commit)
- Path B impl final evidence:`docs/research/2026-05-10-pathB-impl-final-evidence.md`(`c021053`)
- Path B smoke evidence:`docs/research/2026-05-10-pathB-smoke-evidence-LRU-reuse-confirmed.md`(`0198c0d`)
- Wins fill template:`docs/experience/wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`(`1168381`)
- Pipeline runner:`scripts/post_p24_commit_pipeline.sh`
- Validate runner:`scripts/validate_p24_phase0v3.sh`
- Path A KILL precedent:`docs/experience/errors/2026-05-10-37-throughput-bench-killed-pathA-multikey-churn.md`(`e462c53`)
- Codex baseline:`docs/experience/wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`(`0969480`)

## 状态

Decision tree ready for fast license execution post codex Path B commit。
30 min total wall-clock from commit → wins/KILL entry。Tier 1-4 actions
explicit + cross-referenced。Both Tier 1(strong proceed)and Tier 4
(KILL)paths produce **valuable knowledge accumulation** per loop directive
"NULL result 也 commit + push errors entry"。
