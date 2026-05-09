# 3 consecutive ops-layer KILLs — strategic synthesis + anti-pattern #27 candidate

> 2026-05-09 EOD+161 — P1.3 (`edacfe7`)+ P1.4 (`51dd5b2`)+ P1.6 (`4d5f870`)
> 三连 KILL,distinct failure modes。Strategic synthesis 给 future Claude/codex
> 跑下一轮 ops-layer optimization 前必读。

## Three consecutive KILL evidence(matched-machine,N=3 paired bench)

| Attempt | Hypothesis | Failure mode | Evidence |
|---------|-----------|--------------|----------|
| **P1.3** quantized fused_mlp(`edacfe7`)| Launch reduction(4-launch quant fallback → fused gate-up GEMV)| **Saturated kernel**:cuBLAS autotune 已选 optimal algo;launch reduction 不是该 path 真正瓶颈 | TTFT +7.3% regression(env-on 1759.3ms vs env-off 1639.0ms)|
| **P1.4** TileLang FP8 decode wire(`51dd5b2`)| Substrate-existing wire(0 callers in Rust → wire FP8 cubin)| **Substrate semantic mismatch**:TileLang FP8 cubin 的 scale layout / FP8 cast / dequant 与 ARLE 现有 FP8 KV cache 不对齐 | greedy_consistency PASS but output garbage(anti-pattern #26 candidate) |
| **P1.6** QKV packing(`4d5f870`)| Cross-op fusion(3 separate GEMM → 1 packed GEMM)| **Memory-cost-shadow**:packed weight ~1GB overhead → KV pool sizing 缩 → c=4 4k workload 触发 prefix-cache pressure → server stability risk | env-on flat -0.1% TTFT + r3 server failed(43 ok / 88 failed) |

**3 distinct failure modes confirms ops-layer hypothesis space 真正多样**,但
**3 consecutive KILLs at distinct hypothesis types confirms the underlying
optimization frontier 在 current architecture/scope 真正 saturated**。

## Anti-pattern #27 candidate — memory-cost-shadow

新 production-scale failure mode 发现于 P1.6 KILL。

### Failure anatomy

Optimization 尝试增 cache/buffer/weight 副本来 trade memory for launch overhead /
kernel scheduling efficiency。Brief 阶段标 memory cost(e.g., "1GB packed weight"),
但 underestimated **second-order effects** on shared resource sizing:
- KV pool capacity reduced
- Cache prefix-window squeezed
- Prefill admission threshold violated under specific workload

Result:
- Microbench / unit / equivalence / greedy ALL PASS
- Production bench under load(c≥4 4k+ workload)triggers shared-resource pressure
- Server stability fails(timeouts / failed requests)
- Optimization 反 destabilizes the workload it 应优化

### Mitigation framework(proposed)

每个 ops-layer change 增 memory footprint 时,brief 必须 explicitly:
1. 量化 memory delta(LB / MB / GB)
2. **Account for downstream KV pool / cache / scratch budget impact**
3. 设计 KV-pool-pressure bench gate(c=4 4k workload 跑到完成,不 just 120s 截止)
4. Detect server failure patterns(timeouts,失败请求 > X%)→ KILL signal,不算 license

### Codification criteria

Currently 1 production-scale catch(P1.6)。Skill v1.9.0 candidate,等 2nd instance
trigger codification + canonical brief template gate。

监控触发条件:
- 任何 ops change adds > 100 MB memory footprint
- 任何 weight-replication / cache-augmentation hypothesis
- 任何 brief 估 memory cost but absent KV pool sizing 分析

## Strategic synthesis — ops-layer 在 current scope 真正 saturated

### Evidence summary

- **P1.0 LANDED `9773904`** -31.5% TTFT — substrate-existing(W4A8 prefill path),env-gated。这是 **ARLE 当前 architecture/model/workload 下大头优化**,already-realized。
- **P1.2 LANDED `ca0673b`** -13% ITL — graph capture for decode-batched W4A8 path。次大头,already-realized。
- **P1.3 + P1.4 + P1.6 KILL** — 3 distinct hypothesis types,3 distinct failure modes,empirical evidence ops-layer **优化 frontier 在 current scope**(Qwen3-4B / W4A8 / RTX 4090 / c=4 4k workload)真正 saturated。

### Why 在 current scope saturated

- BF16 dense path(包括 fused_mlp,fused_attention)已 production-grade fused
- W4A8 quant path 已 LANDED P1.0 取大头
- Marlin tensor-core utilization 已 well-optimized
- cuBLAS autotune 已选 optimal algo at given shapes
- KV pool sizing 已 tight at c=4 4k(无 headroom for 增 weight cache)
- Substrate semantic alignment(FP8 layout)已被验证 mismatch — 不能简单 wire

### What 还能 attack(post-saturation)

1. **Architectural shift**(scope change):
   - Multi-GPU TP/PP(已有 F0-F8 plan)
   - Model-tier(Qwen3-32B / Qwen4 / DeepSeek-V4)— 改 model 而非 ops
   - Distributed(prefill-decode disaggregation)
2. **Workload-specific tuning**(scope change):
   - Specific ctx lengths(longctx 32k/128k)— 已 active project
   - Specific concurrency patterns(multi-tenant prefix cache,SGLang 强项)
   - Specific request mix(W3/W4 agent workload)— 已 evidence-based
3. **Memory/KV pool restructuring**(non-weight-copy 增量):
   - SGLang-style hierarchical prefix cache
   - HiCache/tier readmission(已有 plan)
   - 但这些不是 "ops-layer optimization",是 system 层 work

### Why P0.0 Phase 1.B SGLang re-verify NOW critical

Without same-machine N=3 paired re-verify of ARLE-vs-SGLang post P1.0+P1.2 wins,
我们 missing **evidence-grade gap measurement**:
- ARLE current TTFT vs SGLang current TTFT(at identical hw + workload)
- Decision tree:
  - 若 ARLE within 5% lead → ops-layer saturated CONFIRMED → architectural pivot
  - 若 ARLE 20%+ gap → architectural shift OR workload-specific tuning needed
  - 若 ARLE 全面 lead → mode-tier pivot(Qwen4 / DSv4)

P0.0 Phase 1.B 是 **strategic decision input**,不是优化 itself。Brief 已 drafted
(`/tmp/codex-brief-p0.0-phase1b-sglang-reverify.txt`),只等用户 green-light。

## Recommendation post 3 KILLs

**Do NOT** continue dispatch ops-layer candidates(#2 RMSNorm+Linear,#3 prep coalescing,
#4 embedding+norm,#5 final logits)without first running P0.0 Phase 1.B。Reason:
- 3 consecutive ops-layer KILLs evidence too strong
- Each ops attempt costs 30-90 min GPU + Claude/codex coordination
- Risk-adjusted ROI 已 dropped — most candidates 也很可能 KILL
- SGLang gap re-verify 是 evidence-driven gate(strategic input not optimization)

**Do**:
1. PushNotification user with 3-KILL summary + recommendation(✅ EOD+161 已发)
2. Wait for user direction:SGLang re-verify dispatch / 切轴 / continue ops anyway
3. Capture anti-pattern #27 candidate research(this entry)
4. Defer remaining 4 ops candidates until SGLang baseline informs decision

## Cross-references

- `edacfe7` P1.3 KILL commit + errors entry
- `51dd5b2` P1.4 KILL commit + errors entry
- `4d5f870` P1.6 KILL commit + errors entry
- `2778dc8` anti-pattern #26 candidate research(same-output-but-garbage)
- `4394899` substrate cleanup audit prep
- `c41198d` greedy_consistency inline doc warning
- `9773904` P1.0 LANDED + `ca0673b` P1.2 LANDED references
- `/tmp/codex-brief-p0.0-phase1b-sglang-reverify.txt` SGLang re-verify brief drafted
- `2e21da1` ops-layer roadmap(historical context)

## Status

3 ops-layer KILL evidence pattern codified。Anti-pattern #27 candidate captured。
Strategic recommendation:**P0.0 Phase 1.B SGLang re-verify**(evidence-driven
gate)before any further ops-layer attempt。

§0 SOLID:`P1.6 KILL` 是 evidence,不是 hypothesis。3 consecutive KILLs distinct
failure modes 是 strong signal,不该被 next ops 尝试 ignore。
